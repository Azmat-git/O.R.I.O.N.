import os
import sys
import glob
import ast
import json
import shutil
import asyncio
import logging
import inspect
import tempfile
import importlib.util
from datetime import datetime
from typing import Any, Callable, get_type_hints
from dotenv import load_dotenv

from telegram import Update, Message
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)
from ollama import AsyncClient

# ==============================================================================
# CONFIGURATION & INITIALISATION
# ==============================================================================
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
AUTHORIZED_USER_ID = int(os.getenv('AUTHORIZED_USER_ID', '0'))
MODEL_NAME = os.getenv('MODEL_NAME', 'mistral-small:24b')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')

# Paramètres d'inférence stricts pour fiabiliser le Tool Calling
OLLAMA_OPTIONS = {
    'temperature': 0.15,      # Réduit drastiquement les hallucinations de paramètres
    'num_ctx': 16384,         # Fenêtre de contexte garantie pour l'historique + outils
}

logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(funcName)s: %(message)s',
    level=logging.INFO
)

ollama_client = AsyncClient(host=OLLAMA_HOST, timeout=180.0)
agent_lock = asyncio.Lock()

# Répertoires de travail
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, 'knowledge')
PLUGINS_DIR = os.path.join(BASE_DIR, 'plugins')
PENDING_PLUGINS_DIR = os.path.join(BASE_DIR, 'plugins_pending')
CREDENTIALS_DIR = os.path.join(BASE_DIR, 'credentials')
MEMORY_FILE = os.path.join(BASE_DIR, 'memory_state.json')

for directory in (KNOWLEDGE_DIR, PLUGINS_DIR, PENDING_PLUGINS_DIR, CREDENTIALS_DIR):
    os.makedirs(directory, exist_ok=True)

# Limites et sécurité
MAX_HISTORY_TURNS = 10
MAX_KNOWLEDGE_CHARS = 5000
MAX_TOOL_OUTPUT_CHARS = 6000
TOOL_EXECUTION_TIMEOUT = 30.0  # Secondes max par appel d'outil
BANNED_MODULES = {'shutil', 'subprocess', 'pty', 'paramiko', 'ctypes'}

# ==============================================================================
# GESTION ATOMIQUE DE LA MÉMOIRE COURT TERME
# ==============================================================================

def load_memory() -> list[dict]:
    """Charge l'historique conversationnel depuis le disque."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            logging.error(f"Erreur de lecture mémoire : {e}")
    return []


def save_memory(history: list[dict]) -> None:
    """Sauvegarde atomique de l'historique pour éviter toute corruption de fichier."""
    try:
        fd, temp_path = tempfile.mkstemp(dir=BASE_DIR, suffix='.tmp')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, MEMORY_FILE)
    except Exception as e:
        logging.error(f"Erreur de sauvegarde atomique mémoire : {e}")


CONVERSATION_HISTORY: list[dict] = load_memory()

# ==============================================================================
# 1. OUTILS CORE (NATIFS)
# ==============================================================================

def search_knowledge(query: str = "") -> str:
    """
    Recherche un mot-clé ou lit un fichier dans le dossier knowledge/.
    Si query est vide, retourne l'index des fichiers disponibles avec leur taille.
    """
    files = sorted(
        glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md")) +
        glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt"))
    )
    if not files:
        return "Le dossier knowledge est actuellement vide."

    if not query or not query.strip():
        index_lines = []
        for f in files:
            size_kb = os.path.getsize(f) / 1024
            index_lines.append(f"- {os.path.basename(f)} ({size_kb:.1f} Ko)")
        return (
            "Index des fichiers disponibles dans knowledge/ :\n" +
            "\n".join(index_lines) +
            "\nAppelle search_knowledge(query='nom_ou_mot_cle') pour lire le contenu."
        )

    results = []
    keywords = [k.lower() for k in query.strip().split() if k]

    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            haystack = f"{filename}\n{content}".lower()
            if any(kw in haystack for kw in keywords):
                results.append(f"=== Fichier : {filename} ===\n{content.strip()}\n")
        except Exception as e:
            results.append(f"Erreur de lecture sur {filename} : {e}")

    if not results:
        return f"Aucun résultat trouvé dans knowledge/ pour '{query}'."

    output = "\n".join(results)
    if len(output) > MAX_KNOWLEDGE_CHARS:
        return output[:MAX_KNOWLEDGE_CHARS] + "\n...[Tronqué : utilise une requête plus spécifique]..."
    return output


def save_note(filename: str, note_content: str, mode: str = "append") -> str:
    """
    Enregistre une note dans knowledge/.
    mode='append' ajoute à la fin avec horodatage ; mode='overwrite' remplace le fichier.
    """
    safe_filename = os.path.basename(filename.strip())
    if not (safe_filename.endswith('.md') or safe_filename.endswith('.txt')):
        safe_filename += ".md"

    target_path = os.path.join(KNOWLEDGE_DIR, safe_filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    write_mode = 'w' if mode == 'overwrite' else 'a'

    try:
        with open(target_path, write_mode, encoding='utf-8') as f:
            if write_mode == 'a':
                f.write(f"\n<!-- Entrée du {timestamp} -->\n{note_content.strip()}\n")
            else:
                f.write(f"<!-- Mis à jour le {timestamp} -->\n{note_content.strip()}\n")
        return f"Succès : Note enregistrée dans '{safe_filename}' (mode={write_mode})."
    except Exception as e:
        return f"Échec d'enregistrement dans '{safe_filename}' : {e}"


def get_current_time() -> str:
    """Retourne la date et l'heure système exactes."""
    return datetime.now().strftime("%A %d %B %Y à %H:%M:%S")


def read_plugin_code(plugin_name: str) -> str:
    """
    Lit le code source d'un plugin (actif ou en quarantaine) pour diagnostiquer un bug
    ou vérifier son implémentation avant de le corriger.
    """
    safe_name = os.path.basename(plugin_name.strip())
    if not safe_name.endswith('.py'):
        safe_name += '.py'

    for folder_label, folder_path in [("plugins (actif)", PLUGINS_DIR), ("plugins_pending (quarantaine)", PENDING_PLUGINS_DIR)]:
        candidate = os.path.join(folder_path, safe_name)
        if os.path.exists(candidate):
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    return f"--- Code source de {safe_name} [{folder_label}] ---\n{f.read()}"
            except Exception as e:
                return f"Erreur de lecture de {safe_name} : {e}"

    return f"Aucun plugin nommé '{safe_name}' trouvé dans plugins/ ou plugins_pending/."


def validate_ast_safety(python_code: str) -> tuple[bool, str]:
    """Vérifie la syntaxe Python, bloque les imports dangereux et vérifie la présence de return."""
    try:
        tree = ast.parse(python_code)
    except SyntaxError as err:
        return False, f"Erreur de syntaxe ligne {err.lineno} : {err.msg}"

    has_public_function = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split('.')[0]
                if root_mod in BANNED_MODULES:
                    return False, f"Sécurité : Import du module '{root_mod}' interdit."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split('.')[0]
                if root_mod in BANNED_MODULES:
                    return False, f"Sécurité : Import depuis '{root_mod}' interdit."
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            has_public_function = True
            has_return = any(isinstance(sub, ast.Return) and sub.value is not None for sub in ast.walk(node))
            if not has_return:
                return (
                    False,
                    f"Qualité : La fonction publique '{node.name}' ne contient aucune instruction 'return <valeur>'. "
                    f"Chaque outil doit retourner une chaîne de caractères (str) exploitable par l'agent."
                )

    if not has_public_function:
        return False, "Qualité : Le plugin doit définir au moins une fonction publique (sans '_' au début)."

    return True, "OK"


def create_plugin(plugin_name: str, python_code: str) -> str:
    """
    Valide l'AST (syntaxe, sécurité, présence de return) et écrit le plugin dans plugins_pending/.
    Requiert l'approbation de Florian via /approve sur Telegram.
    """
    safe_name = os.path.basename(plugin_name.strip())
    if not safe_name.endswith('.py'):
        safe_name += '.py'

    is_valid, reason = validate_ast_safety(python_code)
    if not is_valid:
        return f"Rejet du plugin '{safe_name}' -> {reason}. Corrige le code et rappelle create_plugin."

    filepath = os.path.join(PENDING_PLUGINS_DIR, safe_name)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(python_code)
        return (
            f"Succès : Le plugin '{safe_name}' a été validé (AST + return check) et placé en quarantaine. "
            f"Demande à Florian de l'activer avec la commande : /approve {safe_name}"
        )
    except Exception as e:
        return f"Erreur d'écriture disque : {e}"


CORE_TOOLS: dict[str, Callable] = {
    'search_knowledge': search_knowledge,
    'save_note': save_note,
    'get_current_time': get_current_time,
    'read_plugin_code': read_plugin_code,
    'create_plugin': create_plugin,
}

AVAILABLE_TOOLS: dict[str, Callable] = {}
tools_schema: list[dict] = []

# ==============================================================================
# 2. INTROSPECTION STRICTE & COERCITION DE TYPES
# ==============================================================================

def generate_schema_from_function(func: Callable, module_name: str = "core") -> dict:
    """Génère un schéma JSON complet à partir de la signature et de la docstring Python."""
    sig = inspect.signature(func)
    properties = {}
    required = []

    type_map = {
        str: 'string',
        int: 'integer',
        float: 'number',
        bool: 'boolean',
        list: 'array',
        dict: 'object',
    }

    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls', 'args', 'kwargs'):
            continue

        annotation = param.annotation
        json_type = type_map.get(annotation, 'string')
        desc = f"Paramètre '{param_name}' (type attendu: {json_type})"

        if param.default is not inspect.Parameter.empty:
            desc += f", valeur par défaut: {repr(param.default)}"
        else:
            required.append(param_name)

        properties[param_name] = {
            'type': json_type,
            'description': desc
        }

    doc = inspect.getdoc(func) or f"Outil {func.__name__} ({module_name})"
    return {
        'type': 'function',
        'function': {
            'name': func.__name__,
            'description': doc,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required
            }
        }
    }


def coerce_arguments(func: Callable, raw_args: dict) -> dict:
    """
    Convertit automatiquement les arguments envoyés par le LLM vers les types Python
    annotés dans la signature de la fonction (ex: "5" -> 5, "true" -> True).
    """
    if not isinstance(raw_args, dict):
        return {}

    sig = inspect.signature(func)
    coerced = {}

    for k, v in raw_args.items():
        if k not in sig.parameters:
            continue
        target_type = sig.parameters[k].annotation
        if target_type is inspect.Parameter.empty or v is None:
            coerced[k] = v
            continue

        try:
            if target_type is bool and isinstance(v, str):
                coerced[k] = v.strip().lower() in ('true', '1', 'yes', 'oui', 'vrai')
            elif target_type in (int, float, str) and not isinstance(v, target_type):
                coerced[k] = target_type(v)
            else:
                coerced[k] = v
        except Exception:
            coerced[k] = v

    return coerced


def load_plugins() -> int:
    """Charge les outils Core puis importe proprement tous les plugins actifs."""
    global AVAILABLE_TOOLS, tools_schema
    AVAILABLE_TOOLS = dict(CORE_TOOLS)
    tools_schema = [generate_schema_from_function(fn, "core") for fn in CORE_TOOLS.values()]

    plugin_files = sorted(glob.glob(os.path.join(PLUGINS_DIR, "*.py")))
    loaded_count = 0

    for filepath in plugin_files:
        module_name = os.path.basename(filepath)[:-3]
        if module_name.startswith("__"):
            continue

        try:
            # Purge du cache sys.modules pour garantir un vrai rechargement à chaud
            full_mod_name = f"orion_plugin_{module_name}"
            if full_mod_name in sys.modules:
                del sys.modules[full_mod_name]

            spec = importlib.util.spec_from_file_location(full_mod_name, filepath)
            if not spec or not spec.loader:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[full_mod_name] = module
            spec.loader.exec_module(module)

            if hasattr(module, "TOOLS") and isinstance(module.TOOLS, dict):
                for fn_name, fn_obj in module.TOOLS.items():
                    if callable(fn_obj):
                        AVAILABLE_TOOLS[fn_name] = fn_obj
                        schema = generate_schema_from_function(fn_obj, module_name)
                        schema['function']['name'] = fn_name
                        tools_schema.append(schema)
            else:
                for attr_name, attr in inspect.getmembers(module, inspect.isfunction):
                    if not attr_name.startswith("_") and attr.__module__ == full_mod_name:
                        AVAILABLE_TOOLS[attr_name] = attr
                        tools_schema.append(generate_schema_from_function(attr, module_name))

            loaded_count += 1
            logging.info(f"Plugin actif chargé : {module_name}")
        except Exception as e:
            logging.error(f"Erreur au chargement du plugin {module_name} : {e}")

    return loaded_count


load_plugins()

# ==============================================================================
# 3. DIRECTIVE SYSTÈME
# ==============================================================================
BASE_SYSTEM_PROMPT = """Tu es O.R.I.O.N. (Orchestrateur à Raisonnement Intégré et Opérationnel Neuronal), l'agent IA personnel local de Florian.

Règles fondamentales non négociables :
1. SÉCURITÉ & VIE PRIVÉE : Tu tournes à 100 % en local. Ne tente jamais d'exposer des données privées et protège la confidentialité des fichiers locaux. Les fichiers d'authentification (tokens, JSON OAuth) sont stockés dans le dossier `credentials/`.
2. ALIGNEMENT & ARBITRAGE : Tu agis comme un bras droit loyal dédié exclusivement aux intérêts de Florian. Pour toute action critique ou irréversible, présente les options et demande sa validation explicite avant d'agir.
3. SIMPLICITÉ OPÉRATIONNELLE : Réponds en français natif de façon directe, structurée et concise, sans formules de politesse superflues.
4. ÉVOLUTIVITÉ & AUTO-DIAGNOSTIC : Tu ne modifies jamais `orion.py`. Si un plugin existant échoue ou renvoie un résultat anormal, utilise `read_plugin_code` pour inspecter son code source, puis propose une version corrigée avec `create_plugin`. Chaque fonction d'un plugin DOIT retourner une chaîne `str` explicite contenant les données.
5. MÉMOIRE : Utilise `search_knowledge` dès qu'une demande touche aux projets, comptes ou habitudes de Florian. Utilise `save_note` pour mémoriser toute consigne durable.

Protocole de réponse après appel d'outil :
- Dès qu'un outil te retourne des données (événements d'agenda, e-mails, notes, heure), tu DOIS extraire et présenter ces données concrètes à Florian. Ne réponds JAMAIS par une phrase creuse comme "Opération terminée"."""

# ==============================================================================
# 4. BOUCLE AGENTIQUE MULTI-TOURS AVEC ÉDITION EN DIRECT
# ==============================================================================

async def execute_tool_with_timeout(fn: Callable, raw_args: dict) -> str:
    """Coerce les types, exécute l'outil avec un timeout strict et normalise la sortie en str."""
    safe_args = coerce_arguments(fn, raw_args)

    async def _run():
        if inspect.iscoroutinefunction(fn):
            return await fn(**safe_args)
        return await asyncio.to_thread(fn, **safe_args)

    result = await asyncio.wait_for(_run(), timeout=TOOL_EXECUTION_TIMEOUT)

    if result is None or str(result).strip() == "":
        return (
            f"Avertissement : La fonction '{fn.__name__}' a retourné None ou une chaîne vide. "
            f"Si c'est un plugin, inspecte-le avec read_plugin_code('{fn.__name__}') pour vérifier son 'return'."
        )
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)


async def update_or_send_final(status_msg: Message, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE):
    """Remplace le message de statut par la réponse finale (et découpe si > 4000 caractères)."""
    max_len = 4000
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or ["(Réponse vide)"]

    try:
        await status_msg.edit_text(chunks[0])
    except Exception:
        await context.bot.send_message(chat_id=chat_id, text=chunks[0])

    for extra_chunk in chunks[1:]:
        await context.bot.send_message(chat_id=chat_id, text=extra_chunk)


async def process_agent_task(chat_id: int, user_text: str, status_msg: Message, context: ContextTypes.DEFAULT_TYPE):
    """Boucle ReAct complète : Planification -> Exécution d'outils -> Synthèse garantie."""
    global CONVERSATION_HISTORY

    async with agent_lock:
        try:
            active_tools_list = list(AVAILABLE_TOOLS.keys())
            current_time_str = datetime.now().strftime("%A %d/%m/%Y à %H:%M:%S")

            dynamic_system_prompt = (
                f"{BASE_SYSTEM_PROMPT}\n\n"
                f"--- ÉTAT SYSTÈME ---\n"
                f"- Date et heure : {current_time_str}\n"
                f"- Dossier credentials OAuth/API : {CREDENTIALS_DIR}\n"
                f"- Outils chargés ({len(active_tools_list)}) : {', '.join(active_tools_list)}"
            )

            messages: list[dict[str, Any]] = [{'role': 'system', 'content': dynamic_system_prompt}]
            messages.extend(CONVERSATION_HISTORY)
            messages.append({'role': 'user', 'content': user_text})

            max_iterations = 6
            iteration = 0
            final_text = ""
            tools_trace = []

            generic_phrases = {
                "opération terminée après traitement par les outils.",
                "opération terminée.",
                "tâche terminée.",
                "voici le résultat."
            }

            while iteration < max_iterations:
                iteration += 1
                await context.bot.send_chat_action(chat_id=chat_id, action='typing')

                response = await ollama_client.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools_schema,
                    options=OLLAMA_OPTIONS
                )

                resp_msg = response.message
                content = (resp_msg.content or "").strip()
                tool_calls = resp_msg.tool_calls or []

                # Cas 1 : Aucune demande d'outil -> fin de la boucle
                if not tool_calls:
                    if content and content.lower() not in generic_phrases:
                        final_text = content
                    break

                # Cas 2 : Exécution d'un ou plusieurs outils
                # Sérialisation propre en dict standard pour l'historique Ollama
                serialized_calls = []
                for tc in tool_calls:
                    serialized_calls.append({
                        'function': {
                            'name': tc.function.name,
                            'arguments': dict(tc.function.arguments or {})
                        }
                    })

                messages.append({
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': serialized_calls
                })

                for tc in tool_calls:
                    fn_name = tc.function.name
                    fn_args = dict(tc.function.arguments or {})

                    # Mise à jour visuelle en direct sur Telegram
                    try:
                        await status_msg.edit_text(f"⚙️ Étape {iteration}/{max_iterations} : exécution de `{fn_name}`...")
                    except Exception:
                        pass

                    logging.info(f"[Tour {iteration}] Outil appelé : {fn_name} | Args : {fn_args}")

                    if fn_name in AVAILABLE_TOOLS:
                        try:
                            result = await execute_tool_with_timeout(AVAILABLE_TOOLS[fn_name], fn_args)
                        except asyncio.TimeoutError:
                            result = f"Erreur : L'outil '{fn_name}' a dépassé le délai maximum ({TOOL_EXECUTION_TIMEOUT}s)."
                        except Exception as err:
                            result = f"Erreur d'exécution dans '{fn_name}' ({type(err).__name__}) : {err}"
                    else:
                        result = f"Erreur : Outil '{fn_name}' inconnu. Choisis parmi : {', '.join(active_tools_list)}"

                    if len(result) > MAX_TOOL_OUTPUT_CHARS:
                        result = result[:MAX_TOOL_OUTPUT_CHARS] + "\n...[Sortie tronquée]..."

                    logging.info(f"[Tour {iteration}] Retour de {fn_name} : {result[:200]}...")
                    tools_trace.append(f"{fn_name}({json.dumps(fn_args, ensure_ascii=False)}) => {result[:250]}")

                    messages.append({
                        'role': 'tool',
                        'name': fn_name,
                        'content': result
                    })

            # Synthèse finale forcée si le modèle s'est arrêté après un outil sans conclure proprement
            if not final_text:
                try:
                    await status_msg.edit_text("🧠 Synthèse des résultats en cours...")
                except Exception:
                    pass

                synthesis_messages = list(messages)
                synthesis_messages.append({
                    'role': 'user',
                    'content': (
                        "Exploite les données brutes retournées par les outils ci-dessus et formule maintenant "
                        "ta réponse complète et détaillée pour Florian. Affiche clairement les informations obtenues."
                    )
                })
                final_res = await ollama_client.chat(
                    model=MODEL_NAME,
                    messages=synthesis_messages,
                    options=OLLAMA_OPTIONS
                )
                final_text = (final_res.message.content or "").strip() or "Outils exécutés, mais le modèle n'a généré aucun texte."

            # Mise à jour de la mémoire persistante
            CONVERSATION_HISTORY.append({'role': 'user', 'content': user_text})
            memory_entry = final_text
            if tools_trace:
                memory_entry = f"[Trace outils : {' | '.join(tools_trace)}]\n{final_text}"
            CONVERSATION_HISTORY.append({'role': 'assistant', 'content': memory_entry})

            if len(CONVERSATION_HISTORY) > (MAX_HISTORY_TURNS * 2):
                CONVERSATION_HISTORY = CONVERSATION_HISTORY[-(MAX_HISTORY_TURNS * 2):]

            save_memory(CONVERSATION_HISTORY)
            await update_or_send_final(status_msg, chat_id, final_text, context)

        except Exception as e:
            logging.error(f"Erreur critique boucle agentique : {e}", exc_info=True)
            await update_or_send_final(status_msg, chat_id, f"❌ Erreur interne : {e}", context)


# ==============================================================================
# 5. COMMANDES TELEGRAM & HANDLERS
# ==============================================================================

def is_authorized(update: Update) -> bool:
    if not update.effective_user or update.effective_user.id != AUTHORIZED_USER_ID:
        logging.warning(f"Rejet ID non autorisé : {getattr(update.effective_user, 'id', 'Inconnu')}")
        return False
    return True


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    pending = [os.path.basename(f) for f in glob.glob(os.path.join(PENDING_PLUGINS_DIR, "*.py"))]
    active_plugins = [os.path.basename(f) for f in glob.glob(os.path.join(PLUGINS_DIR, "*.py")) if not os.path.basename(f).startswith("__")]
    msg = (
        f"🛡️ **État du système O.R.I.O.N. v3**\n"
        f"- Modèle : `{MODEL_NAME}` (temp={OLLAMA_OPTIONS['temperature']})\n"
        f"- Mémoire : `{len(CONVERSATION_HISTORY) // 2}/{MAX_HISTORY_TURNS}` tours\n"
        f"- Plugins actifs ({len(active_plugins)}) : `{', '.join(active_plugins) or 'Aucun'}`\n"
        f"- Outils disponibles ({len(AVAILABLE_TOOLS)}) : `{', '.join(AVAILABLE_TOOLS.keys())}`\n"
        f"- En quarantaine ({len(pending)}) : `{', '.join(pending) or 'Aucun'}`"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')


async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Recharge manuellement les plugins à chaud après une modification sur le PC."""
    if not is_authorized(update):
        return
    count = load_plugins()
    await update.message.reply_text(
        f"🔄 Plugins rechargés avec succès ({count} module(s)).\nOutils actifs : `{', '.join(AVAILABLE_TOOLS.keys())}`",
        parse_mode='Markdown'
    )


async def cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    if not context.args:
        await update.message.reply_text("Usage : `/approve nom_plugin.py`", parse_mode='Markdown')
        return

    plugin_name = os.path.basename(context.args[0])
    if not plugin_name.endswith('.py'):
        plugin_name += '.py'

    src = os.path.join(PENDING_PLUGINS_DIR, plugin_name)
    dst = os.path.join(PLUGINS_DIR, plugin_name)

    if not os.path.exists(src):
        await update.message.reply_text(f"❌ `{plugin_name}` introuvable dans `plugins_pending/`.", parse_mode='Markdown')
        return

    shutil.move(src, dst)
    count = load_plugins()
    await update.message.reply_text(
        f"✅ `{plugin_name}` activé ({count} plugins).\nOutils : `{', '.join(AVAILABLE_TOOLS.keys())}`",
        parse_mode='Markdown'
    )


async def cmd_reject(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    if not context.args:
        await update.message.reply_text("Usage : `/reject nom_plugin.py`", parse_mode='Markdown')
        return

    plugin_name = os.path.basename(context.args[0])
    if not plugin_name.endswith('.py'):
        plugin_name += '.py'

    src = os.path.join(PENDING_PLUGINS_DIR, plugin_name)
    if os.path.exists(src):
        os.remove(src)
        await update.message.reply_text(f"🗑️ `{plugin_name}` supprimé de la quarantaine.", parse_mode='Markdown')
    else:
        await update.message.reply_text(f"❌ `{plugin_name}` introuvable.", parse_mode='Markdown')


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY = []
    save_memory(CONVERSATION_HISTORY)
    await update.message.reply_text("🧹 Historique de conversation purgé.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    logging.info(f"Requête reçue : {user_text}")

    status_msg = await context.bot.send_message(
        chat_id=chat_id,
        text="⚙️ Analyse de la requête..."
    )

    asyncio.create_task(process_agent_task(chat_id, user_text, status_msg, context))


# ==============================================================================
# 6. DÉMARRAGE DU SERVICE
# ==============================================================================
if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("approve", cmd_approve))
    app.add_handler(CommandHandler("reject", cmd_reject))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("ORION Core v3 opérationnel.")
    app.run_polling()
