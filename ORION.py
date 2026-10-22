import os
import re
import sys
import glob
import json
import asyncio
import logging
import inspect
import tempfile
import importlib.util
from datetime import datetime
from typing import Any, Callable
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
MODEL_NAME = os.getenv('MODEL_NAME', 'mistral-nemo')
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
CREDENTIALS_DIR = os.path.join(BASE_DIR, 'credentials')
MEMORY_FILE = os.path.join(BASE_DIR, 'memory_state.json')

for directory in (KNOWLEDGE_DIR, PLUGINS_DIR, CREDENTIALS_DIR):
    os.makedirs(directory, exist_ok=True)

# Limites et sécurité
MAX_HISTORY_TURNS = 10
MAX_KNOWLEDGE_CHARS = 5000
MAX_TOOL_OUTPUT_CHARS = 6000
TOOL_EXECUTION_TIMEOUT = 30.0  # Secondes max par appel d'outil

# ==============================================================================
# UTILITAIRE ANTI-TEXTE VIDE (NETTOYAGE UNICODE & TELEGRAM)
# ==============================================================================

# Supprime les caractères invisibles (zero-width space, BOM, etc.) fréquents dans les e-mails
INVISIBLE_CHARS_RE = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]+')


def sanitize_text(val: Any) -> str:
    """Nettoie une chaîne de tous les espaces et caractères invisibles qui font planter Telegram."""
    if val is None:
        return ""
    cleaned = INVISIBLE_CHARS_RE.sub('', str(val))
    return cleaned.strip()


# ==============================================================================
# GESTION ATOMIQUE DE LA MÉMOIRE COURT TERME
# ==============================================================================

def load_memory() -> list[dict]:
    """Charge l'historique conversationnel et purge automatiquement toute entrée vide ou corrompue."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    return []
                valid_history = []
                for item in data:
                    if isinstance(item, dict):
                        role = item.get('role')
                        content = sanitize_text(item.get('content', ''))
                        if role in ('user', 'assistant') and content:
                            valid_history.append({'role': role, 'content': content})
                return valid_history
        except Exception as e:
            logging.error(f"Erreur de lecture mémoire : {e}")
    return []


def save_memory(history: list[dict]) -> None:
    """Sauvegarde atomique de l'historique en garantissant qu'aucun message vide n'est écrit."""
    try:
        clean_history = [
            {'role': m['role'], 'content': sanitize_text(m.get('content', ''))}
            for m in history
            if isinstance(m, dict) and sanitize_text(m.get('content', ''))
        ]
        fd, temp_path = tempfile.mkstemp(dir=BASE_DIR, suffix='.tmp')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(clean_history, f, ensure_ascii=False, indent=2)
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


CORE_TOOLS: dict[str, Callable] = {
    'search_knowledge': search_knowledge,
    'save_note': save_note,
    'get_current_time': get_current_time,
}

AVAILABLE_TOOLS: dict[str, Callable] = {}
tools_schema: list[dict] = []
_PLUGINS_STATE: dict[str, float] = {}

# ==============================================================================
# 2. ARCHITECTURE PLUG-AND-PLAY & INTROSPECTION
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


def _get_plugins_fingerprint() -> dict[str, float]:
    """Retourne un dictionnaire {filepath: mtime} des plugins présents dans plugins/."""
    files = sorted(glob.glob(os.path.join(PLUGINS_DIR, "*.py")))
    return {
        f: os.path.getmtime(f)
        for f in files
        if not os.path.basename(f).startswith("__")
    }


def load_plugins() -> int:
    """
    Charge les outils Core puis branche automatiquement tous les modules déposés dans plugins/.
    Supporte deux formats Plug-and-Play :
    1. Explicite : Le module définit TOOLS (dict) et optionnellement TOOLS_SCHEMA (list).
    2. Automatique : Le module définit simplement des fonctions Python publiques typées.
    """
    global AVAILABLE_TOOLS, tools_schema, _PLUGINS_STATE
    AVAILABLE_TOOLS = dict(CORE_TOOLS)
    tools_schema = [generate_schema_from_function(fn, "core") for fn in CORE_TOOLS.values()]

    plugin_files = sorted(glob.glob(os.path.join(PLUGINS_DIR, "*.py")))
    loaded_count = 0

    for filepath in plugin_files:
        module_name = os.path.basename(filepath)[:-3]
        if module_name.startswith("__"):
            continue

        try:
            full_mod_name = f"orion_plugin_{module_name}"
            if full_mod_name in sys.modules:
                del sys.modules[full_mod_name]

            spec = importlib.util.spec_from_file_location(full_mod_name, filepath)
            if not spec or not spec.loader:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[full_mod_name] = module
            spec.loader.exec_module(module)

            # Format 1 : Le plugin expose explicitement TOOLS (et éventuellement TOOLS_SCHEMA)
            if hasattr(module, "TOOLS") and isinstance(module.TOOLS, dict):
                explicit_schemas = {}
                if hasattr(module, "TOOLS_SCHEMA") and isinstance(module.TOOLS_SCHEMA, list):
                    for s in module.TOOLS_SCHEMA:
                        fn_n = s.get('function', {}).get('name')
                        if fn_n:
                            explicit_schemas[fn_n] = s

                for fn_name, fn_obj in module.TOOLS.items():
                    if callable(fn_obj):
                        AVAILABLE_TOOLS[fn_name] = fn_obj
                        if fn_name in explicit_schemas:
                            tools_schema.append(explicit_schemas[fn_name])
                        else:
                            schema = generate_schema_from_function(fn_obj, module_name)
                            schema['function']['name'] = fn_name
                            tools_schema.append(schema)

            # Format 2 : Plug-and-Play pur (auto-découverte des fonctions publiques du fichier)
            else:
                for attr_name, attr in inspect.getmembers(module, inspect.isfunction):
                    if not attr_name.startswith("_") and attr.__module__ == full_mod_name:
                        AVAILABLE_TOOLS[attr_name] = attr
                        tools_schema.append(generate_schema_from_function(attr, module_name))

            loaded_count += 1
            logging.info(f"Plugin Plug-and-Play actif : {module_name}")
        except Exception as e:
            logging.error(f"Erreur au chargement du plugin {module_name} : {e}")

    _PLUGINS_STATE = _get_plugins_fingerprint()
    return loaded_count


def ensure_plugins_up_to_date() -> None:
    """Vérifie si un fichier a été ajouté, modifié ou supprimé dans plugins/ et recharge à chaud."""
    current_state = _get_plugins_fingerprint()
    if current_state != _PLUGINS_STATE:
        logging.info("Modification détectée dans le dossier plugins/ -> Rechargement automatique à chaud.")
        load_plugins()


# Chargement initial au démarrage
load_plugins()

# ==============================================================================
# 3. DIRECTIVE SYSTÈME
# ==============================================================================
BASE_SYSTEM_PROMPT = """Tu es O.R.I.O.N. (Orchestrateur à Raisonnement Intégré et Opérationnel Neuronal), l'agent IA personnel local de Florian.

Tu opères selon 5 piliers fondamentaux non négociables :

1. SÉCURITÉ, VIE PRIVÉE & EXÉCUTION LOCALE
- Tu tournes à 100 % en local sur le matériel de Florian.
- Ne cherche jamais à exposer des données privées à l'extérieur, n'invente aucun accès distant et préserve l'intégrité des fichiers locaux.

2. ALIGNEMENT STRICT, INTÉRÊT EXCLUSIF & ARBITRAGE (HUMAN-IN-THE-LOOP)
- Tu agis en toutes circonstances pour les intérêts exclusifs de Florian en optimisant son temps et ses ressources.
- RÈGLE DU GARDE-FOU D'ÉCRITURE : Tu as l'interdiction formelle d'exécuter de manière autonome une action modifiant l'état du monde réel ou de ses données (création/modification/suppression d'événements, envoi ou brouillon d'e-mails, suppression de données, transactions).
- PROTOCOLE DE VALIDATION : Pour toute action d'écriture ou de modification, tu dois obligatoirement présenter un récapitulatif clair et complet des paramètres à Florian sur Telegram et attendre son accord explicite au message suivant avant de déclencher l'outil correspondant.

3. INTÉGRITÉ FACTUELLE & VÉRITÉ OPÉRATIONNELLE (ZÉRO HALLUCINATION)
- INTERDICTION D'INVENTER : Tu ne dois sous aucun prétexte simuler, inventer ou deviner le résultat d'un appel d'outil ou d'une API (jamais de faux e-mails, faux identifiants, adresses factices ou listes inventées).
- DÉPENDANCE AU RETOUR OUTIL : Dès qu'une requête nécessite une information extérieure ou système, émets l'appel d'outil officiel et attends impérativement le retour système ('role': 'tool') avant de formuler ta réponse.
- GESTION DU VIDE : Si un outil renvoie une liste vide, une absence de données ou une erreur, constate-le factuellement et sobrement auprès de Florian sans combler les manques par de l'imagination.

4. UTILISATION STRICTE DES OUTILS PLUG-AND-PLAY
- Tu disposes uniquement des outils listés dans ton état système. Tu n'as pas la capacité de créer, coder ou modifier des plugins Python toi-même.
- Si Florian te demande une action pour laquelle aucun outil n'est disponible dans ta liste active, indique-lui simplement quel outil ou connecteur il manque sans tenter de générer du code Python.

5. RIGUEUR TECHNIQUE & AUTO-CORRECTION DES PARAMÈTRES
- RESPECT DES SCHÉMAS D'OUTILS : Vérifie scrupuleusement le nom exact et le typage des paramètres attendus par chaque outil (notamment formats de dates ISO, chaînes, entiers) avant de l'appeler.
- ANALYSE DES ERREURS : En cas de rejet technique (TypeError, ValueError, argument manquant ou exception d'API), lis attentivement le message d'erreur retourné, corrige le nom ou le format de tes arguments et réessaie avec le même outil.
- SIMPLICITÉ & FORMAT TELEGRAM : Formule des réponses directes, concises, structurées et sans bavardage ni politesses superflues."""

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

    if isinstance(result, (dict, list)):
        out_str = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        out_str = str(result or "")

    clean_out = sanitize_text(out_str)
    if not clean_out:
        return f"Information : L'outil '{fn.__name__}' s'est exécuté sans retourner de données."
    return clean_out


async def safe_edit_status(status_msg: Message, text: str) -> None:
    """Met à jour le message de progression sans jamais lever d'exception."""
    clean = sanitize_text(text)
    if not clean:
        return
    try:
        await status_msg.edit_text(clean)
    except Exception as err:
        logging.debug(f"Édition du statut ignorée : {err}")


async def update_or_send_final(status_msg: Message, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE):
    """Remplace le message de statut par la réponse finale avec garantie anti-texte vide."""
    clean_text = sanitize_text(text)
    if not clean_text:
        clean_text = "⚠️ Traitement terminé, mais aucun texte n'a pu être généré."

    max_len = 4000
    chunks = [
        sanitize_text(clean_text[i:i + max_len])
        for i in range(0, len(clean_text), max_len)
    ]
    chunks = [c for c in chunks if c] or ["⚠️ Réponse vide."]

    try:
        await status_msg.edit_text(chunks[0])
    except Exception as edit_err:
        logging.warning(f"Impossible d'éditer status_msg ({edit_err}), envoi d'un nouveau message.")
        try:
            await context.bot.send_message(chat_id=chat_id, text=chunks[0])
        except Exception as send_err:
            logging.error(f"Échec critique d'envoi Telegram : {send_err}")

    for extra_chunk in chunks[1:]:
        try:
            await context.bot.send_message(chat_id=chat_id, text=extra_chunk)
        except Exception as extra_err:
            logging.error(f"Échec d'envoi d'un fragment supplémentaire : {extra_err}")


async def process_agent_task(chat_id: int, user_text: str, status_msg: Message, context: ContextTypes.DEFAULT_TYPE):
    """Boucle ReAct complète : Planification -> Exécution d'outils -> Synthèse garantie."""
    global CONVERSATION_HISTORY

    async with agent_lock:
        try:
            ensure_plugins_up_to_date()

            clean_user_text = sanitize_text(user_text)
            if not clean_user_text:
                await update_or_send_final(status_msg, chat_id, "⚠️ Message reçu vide.", context)
                return

            active_tools_list = list(AVAILABLE_TOOLS.keys())
            current_time_str = datetime.now().strftime("%A %d/%m/%Y à %H:%M:%S")

            dynamic_system_prompt = (
                f"{BASE_SYSTEM_PROMPT}\n\n"
                f"--- ÉTAT SYSTÈME ---\n"
                f"- Date et heure : {current_time_str}\n"
                f"- Outils actifs disponibles ({len(active_tools_list)}) : {', '.join(active_tools_list)}"
            )

            messages: list[dict[str, Any]] = [{'role': 'system', 'content': dynamic_system_prompt}]
            messages.extend(CONVERSATION_HISTORY)
            messages.append({'role': 'user', 'content': clean_user_text})

            max_iterations = 6
            iteration = 0
            final_text = ""
            tools_trace = []
            collected_tool_outputs = []

            generic_phrases = {
                "opération terminée après traitement par les outils.",
                "opération terminée.",
                "tâche terminée.",
                "voici le résultat."
            }

            while iteration < max_iterations:
                iteration += 1
                try:
                    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
                except Exception:
                    pass

                response = await ollama_client.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools_schema,
                    options=OLLAMA_OPTIONS
                )

                resp_msg = response.message
                content = sanitize_text(resp_msg.content)
                tool_calls = resp_msg.tool_calls or []

                # Cas 1 : Aucune demande d'outil -> fin de la boucle
                if not tool_calls:
                    if content and content.lower() not in generic_phrases:
                        final_text = content
                    break

                # Cas 2 : Exécution d'un ou plusieurs outils
                serialized_calls = []
                for tc in tool_calls:
                    fn_n = sanitize_text(getattr(tc.function, 'name', '')) or 'outil_inconnu'
                    raw_args = getattr(tc.function, 'arguments', {}) or {}
                    fn_a = dict(raw_args) if isinstance(raw_args, dict) else {}
                    serialized_calls.append({
                        'function': {
                            'name': fn_n,
                            'arguments': fn_a
                        }
                    })

                messages.append({
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': serialized_calls
                })

                for tc in tool_calls:
                    fn_name = sanitize_text(getattr(tc.function, 'name', '')) or 'outil_inconnu'
                    raw_fn_args = getattr(tc.function, 'arguments', {}) or {}
                    fn_args = dict(raw_fn_args) if isinstance(raw_fn_args, dict) else {}

                    await safe_edit_status(
                        status_msg,
                        f"⚙️ Étape {iteration}/{max_iterations} : exécution de {fn_name}..."
                    )

                    logging.info(f"[Tour {iteration}] Outil appelé : {fn_name} | Args : {fn_args}")

                    if fn_name in AVAILABLE_TOOLS:
                        try:
                            result = await execute_tool_with_timeout(AVAILABLE_TOOLS[fn_name], fn_args)
                        except asyncio.TimeoutError:
                            result = f"Erreur : L'outil '{fn_name}' a dépassé le délai maximum ({TOOL_EXECUTION_TIMEOUT}s)."
                        except Exception as err:
                            sig = inspect.signature(AVAILABLE_TOOLS[fn_name])
                            result = (
                                f"Erreur d'exécution dans '{fn_name}' ({type(err).__name__}) : {err}. "
                                f"Signature attendue : {fn_name}{sig}"
                            )
                    else:
                        result = f"Erreur : Outil '{fn_name}' inexistant. Outils disponibles : {', '.join(active_tools_list)}"

                    result = sanitize_text(result) or f"Outil {fn_name} exécuté sans retour."

                    if len(result) > MAX_TOOL_OUTPUT_CHARS:
                        result = result[:MAX_TOOL_OUTPUT_CHARS] + "\n...[Sortie tronquée]..."

                    logging.info(f"[Tour {iteration}] Retour de {fn_name} : {result[:200]}...")
                    tools_trace.append(f"{fn_name}({json.dumps(fn_args, ensure_ascii=False)}) => {result[:250]}")
                    collected_tool_outputs.append(f"--- Retour de {fn_name} ---\n{result}")

                    messages.append({
                        'role': 'tool',
                        'content': f"[Résultat de l'outil {fn_name}]\n{result}"
                    })

            # Synthèse finale si le modèle a renvoyé une chaîne vide après les outils
            clean_final = sanitize_text(final_text)

            if not clean_final and collected_tool_outputs:
                await safe_edit_status(status_msg, "🧠 Synthèse des résultats en cours...")

                joined_outputs = "\n\n".join(collected_tool_outputs)
                synthesis_prompt = [
                    {'role': 'system', 'content': BASE_SYSTEM_PROMPT},
                    {
                        'role': 'user',
                        'content': (
                            f"Demande initiale de Florian : {clean_user_text}\n\n"
                            f"Voici les données réelles retournées par les outils :\n{joined_outputs}\n\n"
                            f"Rédige maintenant une réponse claire, complète et directe en français pour Florian "
                            f"à partir de ces données."
                        )
                    }
                ]
                try:
                    final_res = await ollama_client.chat(
                        model=MODEL_NAME,
                        messages=synthesis_prompt,
                        options=OLLAMA_OPTIONS
                    )
                    clean_final = sanitize_text(final_res.message.content)
                except Exception as synth_err:
                    logging.warning(f"Échec de la synthèse : {synth_err}")

            # FILET DE SÉCURITÉ ABSOLU : Si le LLM reste muet, on affiche directement le retour des outils
            if not clean_final:
                if collected_tool_outputs:
                    clean_final = "📋 Données récupérées :\n\n" + "\n\n".join(collected_tool_outputs)
                else:
                    clean_final = "⚠️ Aucune réponse générée par le modèle."

            final_text = clean_final

            # Mise à jour propre de la mémoire persistante (jamais de chaîne vide)
            CONVERSATION_HISTORY.append({'role': 'user', 'content': clean_user_text})
            memory_entry = final_text
            if tools_trace:
                memory_entry = f"[Trace outils : {' | '.join(tools_trace)}]\n{final_text}"
            CONVERSATION_HISTORY.append({'role': 'assistant', 'content': sanitize_text(memory_entry)})

            if len(CONVERSATION_HISTORY) > (MAX_HISTORY_TURNS * 2):
                CONVERSATION_HISTORY = CONVERSATION_HISTORY[-(MAX_HISTORY_TURNS * 2):]

            save_memory(CONVERSATION_HISTORY)
            await update_or_send_final(status_msg, chat_id, final_text, context)

        except Exception as e:
            logging.error(f"Erreur critique boucle agentique : {e}", exc_info=True)
            err_str = sanitize_text(f"❌ Erreur interne : {type(e).__name__} - {e}")
            await update_or_send_final(status_msg, chat_id, err_str or "❌ Erreur interne inconnue.", context)


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
    ensure_plugins_up_to_date()
    active_plugins = [
        os.path.basename(f)
        for f in sorted(glob.glob(os.path.join(PLUGINS_DIR, "*.py")))
        if not os.path.basename(f).startswith("__")
    ]
    msg = (
        f"🛡️ **État du système O.R.I.O.N. v3 (Plug-and-Play)**\n"
        f"- Modèle : `{MODEL_NAME}` (temp={OLLAMA_OPTIONS['temperature']})\n"
        f"- Mémoire : `{len(CONVERSATION_HISTORY) // 2}/{MAX_HISTORY_TURNS}` tours\n"
        f"- Modules plugins ({len(active_plugins)}) : `{', '.join(active_plugins) or 'Aucun'}`\n"
        f"- Outils actifs ({len(AVAILABLE_TOOLS)}) : `{', '.join(AVAILABLE_TOOLS.keys())}`"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')


async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force manuellement le rechargement de tous les plugins du dossier plugins/."""
    if not is_authorized(update):
        return
    count = load_plugins()
    await update.message.reply_text(
        f"🔄 Plugins rechargés ({count} module(s)).\nOutils actifs : `{', '.join(AVAILABLE_TOOLS.keys())}`",
        parse_mode='Markdown'
    )


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

    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = sanitize_text(update.message.text)
    if not user_text:
        return

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
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("ORION Core v3 (Plug-and-Play) opérationnel.")
    app.run_polling()
