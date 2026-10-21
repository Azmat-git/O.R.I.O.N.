import os
import glob
import ast
import asyncio
import logging
import inspect
import importlib.util
from datetime import datetime
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from ollama import Client

# ==============================================================================
# CONFIGURATION & INITIALISATION
# ==============================================================================
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
AUTHORIZED_USER_ID = int(os.getenv('AUTHORIZED_USER_ID', '0'))
MODEL_NAME = os.getenv('MODEL_NAME', 'mistral-small:24b')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

ollama_client = Client(host='http://127.0.0.1:11434', timeout=None)

# Répertoires de travail de l'agent
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, 'knowledge')
PLUGINS_DIR = os.path.join(BASE_DIR, 'plugins')

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(PLUGINS_DIR, exist_ok=True)

# Mémoire conversationnelle glissante (court terme)
CONVERSATION_HISTORY = []
MAX_HISTORY_TURNS = 10

# ==============================================================================
# 1. OUTILS CORE (TOOLS)
# ==============================================================================

def search_knowledge(query: str = "") -> str:
    """Parcourt les fichiers du dossier knowledge et lit les informations stockées."""
    files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md")) + glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt"))
    if not files:
        return "Le dossier de connaissances est actuellement vide."

    results = []
    query_lower = query.lower()

    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if not query or query_lower in content.lower() or query_lower in filename.lower():
                    results.append(f"--- Fichier: {filename} ---\n{content}\n")
        except Exception as e:
            results.append(f"Erreur de lecture sur {filename} : {e}")

    if not results:
        return f"Aucune information trouvée pour : '{query}'"

    return "\n".join(results)


def save_note(filename: str, note_content: str) -> str:
    """Enregistre ou ajoute une information dans un fichier du dossier knowledge."""
    safe_filename = os.path.basename(filename)
    if not (safe_filename.endswith('.md') or safe_filename.endswith('.txt')):
        safe_filename += ".md"

    target_path = os.path.join(KNOWLEDGE_DIR, safe_filename)
    try:
        with open(target_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{note_content.strip()}\n")
        return f"Note enregistrée avec succès dans {safe_filename}."
    except Exception as e:
        return f"Échec de l'enregistrement : {e}"


def get_current_time() -> str:
    """Retourne la date et l'heure système actuelles de la machine."""
    return datetime.now().strftime("%A %d %B %Y à %H:%M:%S")


def create_plugin(plugin_name: str, python_code: str) -> str:
    """Valide la syntaxe, écrit le module dans plugins/ et l'enregistre à chaud dans le Core."""
    safe_name = os.path.basename(plugin_name)
    if not safe_name.endswith('.py'):
        safe_name += '.py'

    # Vérification syntaxique avant écriture sur disque
    try:
        ast.parse(python_code)
    except SyntaxError as err:
        return f"Erreur de syntaxe Python détectée : {err}. Corrige le code et réessaie."

    filepath = os.path.join(PLUGINS_DIR, safe_name)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(python_code)

        # Rechargement dynamique immédiat des plugins
        loaded_count = load_plugins()
        return f"Succès : Le plugin {safe_name} a été validé, écrit et activé dans le Core ({loaded_count} plugin(s) actif(s))."
    except Exception as e:
        return f"Erreur système d'écriture : {e}"


# Outils natifs de base
CORE_TOOLS = {
    'search_knowledge': search_knowledge,
    'save_note': save_note,
    'create_plugin': create_plugin,
    'get_current_time': get_current_time,
}

AVAILABLE_TOOLS = dict(CORE_TOOLS)

CORE_TOOLS_SCHEMA = [
    {
        'type': 'function',
        'function': {
            'name': 'search_knowledge',
            'description': 'Recherche et lit les documents, fiches et notes stockés localement.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Mot-clé ou sujet à rechercher (laisser vide pour tout lire).'
                    }
                },
                'required': []
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'save_note',
            'description': 'Ajoute une note ou une information mémorisée dans le dossier knowledge.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'filename': {
                        'type': 'string',
                        'description': 'Nom du fichier cible (ex: journal.md, directives.md).'
                    },
                    'note_content': {
                        'type': 'string',
                        'description': 'Contenu exact du texte à enregistrer.'
                    }
                },
                'required': ['filename', 'note_content']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': "Consulte l'horloge système du PC pour obtenir la date, le jour et l'heure actuels.",
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'create_plugin',
            'description': 'Écrit un script Python valide dans le dossier plugins pour ajouter une nouvelle compétence.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'plugin_name': {
                        'type': 'string',
                        'description': 'Nom du fichier Python (ex: mon_outil.py).'
                    },
                    'python_code': {
                        'type': 'string',
                        'description': 'Code Python source complet et fonctionnel.'
                    }
                },
                'required': ['plugin_name', 'python_code']
            }
        }
    }
]

tools_schema = list(CORE_TOOLS_SCHEMA)

# ==============================================================================
# 2. CHARGEMENT DYNAMIQUE DES PLUGINS
# ==============================================================================

def load_plugins() -> int:
    """Scanne le dossier plugins/, importe les modules et met à jour AVAILABLE_TOOLS et tools_schema."""
    global AVAILABLE_TOOLS, tools_schema
    AVAILABLE_TOOLS = dict(CORE_TOOLS)
    tools_schema = list(CORE_TOOLS_SCHEMA)

    plugin_files = glob.glob(os.path.join(PLUGINS_DIR, "*.py"))
    loaded_count = 0

    for filepath in plugin_files:
        module_name = os.path.basename(filepath)[:-3]
        if module_name.startswith("__"):
            continue

        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if not spec or not spec.loader:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Option A : Le plugin exporte explicitement TOOLS et TOOLS_SCHEMA
            if hasattr(module, "TOOLS") and isinstance(module.TOOLS, dict):
                AVAILABLE_TOOLS.update(module.TOOLS)

                if hasattr(module, "TOOLS_SCHEMA") and isinstance(module.TOOLS_SCHEMA, list):
                    for schema in module.TOOLS_SCHEMA:
                        if schema not in tools_schema:
                            tools_schema.append(schema)

            # Option B : Auto-découverte des fonctions publiques si pas de dictionnaire TOOLS
            else:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if inspect.isfunction(attr) and not attr_name.startswith("_"):
                        AVAILABLE_TOOLS[attr_name] = attr
                        schema = {
                            'type': 'function',
                            'function': {
                                'name': attr_name,
                                'description': inspect.getdoc(attr) or f"Outil du module {module_name}",
                                'parameters': {
                                    'type': 'object',
                                    'properties': {},
                                    'required': []
                                }
                            }
                        }
                        if schema not in tools_schema:
                            tools_schema.append(schema)

            loaded_count += 1
            logging.info(f"Plugin chargé avec succès : {module_name}")

        except Exception as e:
            logging.error(f"Échec du chargement du module {module_name} : {e}")

    return loaded_count

# Premier chargement à l'initialisation du service
load_plugins()

# ==============================================================================
# 3. DIRECTIVE SYSTÈME
# ==============================================================================
BASE_SYSTEM_PROMPT = """Tu es O.R.I.O.N. (Orchestrateur à Raisonnement Intégré et Opérationnel Neuronal), l'agent IA personnel local de Florian.

Tu opères selon 5 règles fondamentales non négociables :
1. SÉCURITÉ & VIE PRIVÉE : Tu tournes à 100 % en local sur son matériel. Ne tente jamais d'exposer des données privées, n'invente pas d'accès externe et protège la confidentialité des fichiers locaux.
2. ALIGNEMENT, INTÉRÊT EXCLUSIF & ARBITRAGE : Tu agis en toutes circonstances comme un bras droit loyal dédié exclusivement aux intérêts de Florian. Priorise systématiquement son temps, sa sécurité numérique, l'optimisation de ses ressources et la fiabilité de ses décisions. Tu n'engages jamais d'action critique, irréversible ou stratégique (dépense financière, suppression de données personnelles, modification d'accès ou choix impactant ses systèmes) de manière autonome : pour toute décision importante, présente clairement les options, les risques, et demande obligatoirement sa validation explicite avant d'agir.
3. SIMPLICITÉ OPÉRATIONNELLE : Florian interagit avec toi via Telegram. Tes réponses doivent être directes, concises, structurées et rédigées en français natif, sans formules de politesse superflues ni bavardage.
4. ÉVOLUTIVITÉ MODULAIRE : Tu ne touches jamais au fichier cœur (orion.py). Toute nouvelle compétence technique doit être isolée sous forme de module dans le dossier plugins/ via l'outil create_plugin.
5. AUTO-AMÉLIORATION & MÉMOIRE : Tu t'alimentes de façon autonome sur les données de Florian. Consulte tes connaissances locales avec search_knowledge dès qu'une question porte sur ses projets, directives ou habitudes. Quand Florian te donne une consigne durable, un retour ou une information à retenir, utilise save_note pour mettre à jour ta mémoire.

Comportement agentique attendu :
- Tu disposes d'une boucle de réflexion autonome. Si l'exécution d'un outil renvoie une erreur ou un rejet, analyse le retour, adapte tes paramètres ou corrige ton code, et réessaie immédiatement sans abandonner.
- Sois factuel, rigoureux et précis."""

# ==============================================================================
# 4. BOUCLE AGENTIQUE MULTI-TOURS (FEEDBACK LOOP)
# ==============================================================================

async def process_agent_task(chat_id: int, user_text: str, context: ContextTypes.DEFAULT_TYPE):
    """Gère la boucle agentique ReAct avec mémoire, plugins dynamiques et synthèse finale."""
    global CONVERSATION_HISTORY

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')

        # État système injecté dynamiquement
        active_tools_list = list(AVAILABLE_TOOLS.keys())
        current_time_str = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")

        dynamic_system_prompt = (
            f"{BASE_SYSTEM_PROMPT}\n\n"
            f"--- ÉTAT ACTUEL DU SYSTÈME ---\n"
            f"- Horodatage courant : {current_time_str}\n"
            f"- Outils actuellement actifs et exécutables : {active_tools_list}"
        )

        messages = [{'role': 'system', 'content': dynamic_system_prompt}]
        messages.extend(CONVERSATION_HISTORY)
        messages.append({'role': 'user', 'content': user_text})

        max_iterations = 5
        iteration = 0
        final_text = ""

        while iteration < max_iterations:
            iteration += 1
            await context.bot.send_chat_action(chat_id=chat_id, action='typing')

            response = await asyncio.to_thread(
                ollama_client.chat,
                model=MODEL_NAME,
                messages=messages,
                tools=tools_schema
            )

            response_msg = response.message

            # Cas 1 : L'agent formule une réponse textuelle directe sans appel d'outil
            if not response_msg.tool_calls:
                final_text = response_msg.content
                break

            # Cas 2 : L'agent appelle un ou plusieurs outils
            messages.append(response_msg)

            for tool_call in response_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments

                logging.info(f"[Boucle agentique - Tour {iteration}] Appel outil : {fn_name} | Paramètres : {fn_args}")

                if fn_name in AVAILABLE_TOOLS:
                    try:
                        result = AVAILABLE_TOOLS[fn_name](**fn_args)
                    except Exception as err:
                        result = f"Erreur d'exécution interne : {err}"
                else:
                    result = f"Erreur : L'outil '{fn_name}' n'est pas accessible."

                messages.append({
                    'role': 'tool',
                    'content': str(result)
                })

        # Si le modèle a enchaîné des outils sans formuler sa synthèse finale :
        if not final_text:
            synthesis_prompt = list(messages)
            synthesis_prompt.append({
                'role': 'user',
                'content': 'Formule maintenant ta réponse finale claire et synthétique pour répondre à ma demande initiale en exploitant les données des outils exécutés.'
            })
            final_res = await asyncio.to_thread(
                ollama_client.chat,
                model=MODEL_NAME,
                messages=synthesis_prompt
            )
            final_text = final_res.message.content

        # Enregistrement dans la mémoire conversationnelle glissante
        CONVERSATION_HISTORY.append({'role': 'user', 'content': user_text})
        CONVERSATION_HISTORY.append({'role': 'assistant', 'content': final_text})
        if len(CONVERSATION_HISTORY) > (MAX_HISTORY_TURNS * 2):
            CONVERSATION_HISTORY = CONVERSATION_HISTORY[-(MAX_HISTORY_TURNS * 2):]

        await context.bot.send_message(chat_id=chat_id, text=final_text)

    except Exception as e:
        logging.error(f"Erreur critique dans la boucle agentique : {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"Erreur interne : {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Règle 1 : Rejet des utilisateurs non autorisés
    if update.effective_user.id != AUTHORIZED_USER_ID:
        logging.warning(f"Message ignoré - ID non autorisé : {update.effective_user.id}")
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    logging.info(f"Message reçu de Florian : {user_text}")

    # 1. Accusé de réception synchrone immédiat
    await context.bot.send_message(
        chat_id=chat_id,
        text="⚙️ Reçu. Analyse et exécution en cours..."
    )

    # 2. Lancement asynchrone non-bloquant de la tâche agentique
    asyncio.create_task(process_agent_task(chat_id, user_text, context))


# ==============================================================================
# 5. DÉMARRAGE DU BOT
# ==============================================================================
if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("ORION Core actif, chargeur de plugins dynamique et mémoire prêts.")
    app.run_polling()
