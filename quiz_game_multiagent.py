import pandas as pd
import logging
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import random

# ------------------------------------------------------------
# Dependência opcional para rodar Llama‑3 localmente
# ------------------------------------------------------------
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # fallback mock

DEFAULT_MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"  # pode ser alterado via --model
MODEL_CTX = 4096
TOTAL_ROUNDS = 20  # número máximo de perguntas por partida

# ------------------------------------------------------------
# Utilitário LLM
# ------------------------------------------------------------

def load_llm(model_path: str):
    if Llama is None:
        logging.warning("llama-cpp-python não instalado — respostas serão simuladas.")
        return None

    if not Path(model_path).is_file():
        logging.warning(f"Modelo não encontrado em '{model_path}'. Rodando em modo stub.")
        return None

    return Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=MODEL_CTX,
        verbose=False        # <<< silencia a saída do modelo
    )
# ------------------------------------------------------------
# Estruturas de dados
# ------------------------------------------------------------
@dataclass
class Question:
    id: int
    enunciado: str
    alternativas: Dict[str, str]
    resposta: str
    dificuldade: int
    tema: str = ""


@dataclass
class Benefit:
    descricao: str
    quantidade: int


@dataclass
class Message:
    sender: str
    receiver: str
    content: str

# ------------------------------------------------------------
# Barramento de Mensagens
# ------------------------------------------------------------
class MessageBus:
    def __init__(self):
        self.log: List[Message] = []
        self.agents: Dict[str, "BaseAgent"] = {}

    def register(self, agent: "BaseAgent"):
        self.agents[agent.name] = agent

    def send(self, sender: str, receiver: str, content: str):
        msg = Message(sender, receiver, content)
        self.log.append(msg)
        if receiver in self.agents:
            self.agents[receiver].receive(msg)
        else:
            logging.error(f"Agente {receiver} não registrado.")

    def broadcast(self, sender: str, receivers: List[str], content: str):
        for r in receivers:
            self.send(sender, r, content)

    def conversation_log(self):
        return self.log

# ------------------------------------------------------------
# Agente Base
# ------------------------------------------------------------
class BaseAgent:
    def __init__(self, name: str, bus: MessageBus, llm):
        self.name = name
        self.bus = bus
        self.bus.register(self)
        self.llm = llm

    def _ask_llm(self, prompt: str, max_tokens: int = 256):
        if self.llm is None:
            return "(resposta mock)"
        return self.llm(prompt=prompt, max_tokens=max_tokens, stop=["\n"]).strip()

    def receive(self, message: Message):
        raise NotImplementedError

# ------------------------------------------------------------
# Agente de Questões
# ------------------------------------------------------------
class QuestionAgent(BaseAgent):
    def __init__(self, bus: MessageBus, llm, questions: List[Question]):
        super().__init__("Questions", bus, llm)
        self.questions = {q.id: q for q in questions}
        # Group questions by theme for random selection
        self.questions_by_theme = {}
        for q in questions:
            if q.tema not in self.questions_by_theme:
                self.questions_by_theme[q.tema] = []
            self.questions_by_theme[q.tema].append(q)

    def get_random_theme(self) -> str:
        """Returns a random theme from available themes."""
        return random.choice(list(self.questions_by_theme.keys()))

    def format_alternatives(self, alternatives: Dict[str, str]) -> str:
        """Format alternatives in a readable way."""
        return "\n".join(f"{key}) {value}" for key, value in sorted(alternatives.items()))

    def escape_special_chars(self, text: str) -> str:
        """Escape special characters that might interfere with message parsing."""
        return text.replace("{", "{{").replace("}", "}}").replace("%", "%%")

    def get_question_by_strategy(self, qid: int, required_difficulty: int, required_theme: str = "") -> Question:
        """Get a question that matches the strategy requirements."""
        # If the question ID is 1 or 2, force difficulty to 1
        if qid in [1, 2]:
            required_difficulty = 1

        # If theme is empty, pick a random theme
        if not required_theme:
            required_theme = self.get_random_theme()
        elif required_theme == "diferente":
            # Pick a different theme than the current question's theme
            if qid in self.questions:
                current_theme = self.questions[qid].tema
                available_themes = [t for t in self.questions_by_theme.keys() if t != current_theme]
                if available_themes:
                    required_theme = random.choice(available_themes)

        # Try to find a matching question
        if qid in self.questions:
            q = self.questions[qid]
            if q.dificuldade == required_difficulty and (not required_theme or q.tema == required_theme):
                return q

        # If no matching question found, find an alternative with matching difficulty and theme
        matching_questions = [
            q for q in self.questions.values()
            if q.dificuldade == required_difficulty and (not required_theme or q.tema == required_theme)
        ]
        
        if matching_questions:
            return random.choice(matching_questions)
        
        # If still no match, return the original question as fallback
        return self.questions[qid]

    def receive(self, message: Message):
        if message.content.startswith("GET_QUESTION"):
            parts = message.content.split()
            qid = int(parts[1])
            difficulty = int(parts[2])
            theme = parts[3] if len(parts) > 3 else ""
            
            q = self.get_question_by_strategy(qid, difficulty, theme)
            if not q:
                self.bus.send(self.name, message.sender, f"QUESTION_NOT_FOUND {qid}")
                return
            
            # Escape special characters and format question data
            question_text = self.escape_special_chars(q.enunciado)
            alternatives = {k: self.escape_special_chars(v) for k, v in q.alternativas.items()}
            alternatives_str = self.format_alternatives(alternatives)
            
            # Send current question info to BenefitAgent
            self.bus.send(self.name, "Benefits", f"CURRENT_QUESTION {q.id} {q.resposta}")
            
            # Use a special delimiter to separate question parts
            self.bus.send(self.name, message.sender,
                       f"QUESTION_DATA {q.id}|||{question_text}|||{alternatives_str}")

        elif message.content.startswith("CHECK_ANSWER"):
            _, qid, guess = message.content.split()
            qid = int(qid)
            q = self.questions.get(qid)
            if not q:
                self.bus.send(self.name, message.sender, f"QUESTION_NOT_FOUND {qid}")
                return
            result_str = "CORRECT" if guess.upper() == q.resposta.upper() else "WRONG"
            self.bus.broadcast(
                self.name,
                ["Console", "Strategy"],
                f"ANSWER_RESULT {qid} {result_str} {q.resposta}"
            )

# ------------------------------------------------------------
# Agente de Benefícios
# ------------------------------------------------------------
class BenefitAgent(BaseAgent):
    def __init__(self, bus: MessageBus, llm, benefits: Dict[str, Benefit]):
        super().__init__("Benefits", bus, llm)
        self.benefits = benefits
        self.benefit_descriptions = {
            "Ajuda dos universitários": "Eles votam na alternativa que acreditam ser a correta",
            "Cartas": "Elimina 2 alternativas incorretas",
            "Pular": "Pular uma questão",
            "Mudar": "Mudar a pergunta para um tema escolhido"
        }
        self.current_question = None
        self.current_answer = None

    def _simulate_university_help(self) -> str:
        """Simula a ajuda dos universitários."""
        if not self.current_answer:
            return "Desculpe, não tenho a resposta para ajudar!"
        
        # 70% chance de acertar
        if random.random() < 0.7:
            correct = self.current_answer
            # Distribuir votos com maior peso para a resposta correta
            votes = {
                correct: random.randint(45, 60),
                self._get_random_wrong_alternative(): random.randint(15, 25),
                self._get_random_wrong_alternative(): random.randint(10, 20),
                self._get_random_wrong_alternative(): random.randint(5, 15)
            }
        else:
            # 30% chance de errar - distribuir votos aleatoriamente
            alternatives = ['A', 'B', 'C', 'D']
            votes = {alt: random.randint(10, 40) for alt in alternatives}
        
        # Formatar resultado
        result = "\nVotação dos universitários:\n"
        for alt, vote in sorted(votes.items()):
            result += f"{alt}: {vote}%\n"
        return result

    def _get_random_wrong_alternative(self) -> str:
        """Retorna uma alternativa errada aleatória."""
        alternatives = ['A', 'B', 'C', 'D']
        wrong_alternatives = [alt for alt in alternatives if alt != self.current_answer]
        return random.choice(wrong_alternatives)

    def _eliminate_wrong_alternatives(self) -> str:
        """Elimina duas alternativas erradas."""
        if not self.current_answer:
            return "Desculpe, não tenho a resposta para ajudar!"
        
        alternatives = ['A', 'B', 'C', 'D']
        wrong_alternatives = [alt for alt in alternatives if alt != self.current_answer]
        eliminated = random.sample(wrong_alternatives, 2)
        
        result = "\nAlternativas eliminadas:\n"
        for alt in sorted(eliminated):
            result += f"{alt}\n"
        return result

    def receive(self, message: Message):
        if message.content == "GET_AVAILABLE_BENEFITS":
            data = {k: v.quantidade for k, v in self.benefits.items() if v.quantidade > 0}
            self.bus.send(self.name, message.sender, f"AVAILABLE_BENEFITS {data}")

        elif message.content.startswith("CURRENT_QUESTION"):
            _, qid, answer = message.content.split()
            self.current_question = qid
            self.current_answer = answer

        elif message.content.startswith("USE_BENEFIT"):
            _, key = message.content.split(None, 1)
            # Extract just the benefit name without the description
            benefit_key = key.split(" (")[0] if " (" in key else key
            b = self.benefits.get(benefit_key)
            
            if b and b.quantidade > 0:
                b.quantidade -= 1
                description = self.benefit_descriptions.get(benefit_key, "")
                
                # Implementar o efeito de cada benefício
                effect = ""
                if benefit_key == "Ajuda dos universitários":
                    effect = self._simulate_university_help()
                elif benefit_key == "Cartas":
                    effect = self._eliminate_wrong_alternatives()
                
                self.bus.send(self.name, message.sender, 
                            f"BENEFIT_USED {benefit_key} ({description}) {b.quantidade}\n{effect}")
            else:
                self.bus.send(self.name, message.sender, f"BENEFIT_NOT_AVAILABLE {benefit_key}")

# ------------------------------------------------------------
# Agente de Estratégia (ordem definida por CSV)
# ------------------------------------------------------------
class StrategyAgent(BaseAgent):
    def __init__(self, bus: MessageBus, llm, sequence: List[Dict[str, any]]):
        super().__init__("Strategy", bus, llm)
        self.sequence = sequence
        self.idx = 0
        self.score = 0
        self.used_benefits: List[str] = []
        self.game_over = False

    def start(self):
        if not self.sequence:
            self.bus.send(self.name, "Console", "GAME_OVER Sequência vazia")
            return
        current = self.sequence[0]
        self.bus.send(
            self.name, 
            "Questions", 
            f"GET_QUESTION {current['id']} {current['difficulty']} {current['theme']}"
        )

    def receive(self, message: Message):
        if message.content == "GAME_OVER":
            self.game_over = True
            self.bus.send(self.name, "Console", f"GAME_OVER Score final: {self.score}/{len(self.sequence)}")
            return

        if message.content.startswith("QUESTION_DATA"):
            # repassa apenas para o Console
            self.bus.send(self.name, "Console", message.content)

        elif message.content.startswith("ANSWER_RESULT"):
            _, qid, result, correct = message.content.split()
            if result == "CORRECT":
                self.score += 1
                # próxima pergunta
                self.idx += 1
                if self.idx < len(self.sequence) and not self.game_over:
                    current = self.sequence[self.idx]
                    self.bus.send(
                        self.name, 
                        "Questions", 
                        f"GET_QUESTION {current['id']} {current['difficulty']} {current['theme']}"
                    )
                elif self.idx >= len(self.sequence):
                    # Only end if we've completed all questions
                    self.game_over = True
                    self.bus.send(self.name, "Console", f"GAME_OVER Score final: {self.score}/{len(self.sequence)}")
            else:
                # Game over on wrong answer
                self.game_over = True
                self.bus.send(self.name, "Console", f"GAME_OVER Score final: {self.score}/{len(self.sequence)}")

        elif message.content.startswith("BENEFIT_USED"):
            _, benefit = message.content.split(" ", 1)
            benefit_key = benefit.split(" ")[0]  # Get just the first word of the benefit
            if benefit_key not in self.used_benefits:
                self.used_benefits.append(benefit_key)
                # Handle specific benefits
                if benefit_key == "Pular":
                    self.idx += 1
                    if self.idx < len(self.sequence):
                        current = self.sequence[self.idx]
                        self.bus.send(
                            self.name, 
                            "Questions", 
                            f"GET_QUESTION {current['id']} {current['difficulty']} {current['theme']}"
                        )
                    else:
                        self.game_over = True
                        self.bus.send(self.name, "Console", f"GAME_OVER Score final: {self.score}/{len(self.sequence)}")
                elif benefit_key == "Mudar":
                    # Stay on same question but request a new one with different theme
                    if self.idx < len(self.sequence):
                        current = self.sequence[self.idx]
                        self.bus.send(
                            self.name, 
                            "Questions", 
                            f"GET_QUESTION {current['id']} {current['difficulty']} diferente"
                        )

# ------------------------------------------------------------
# Agente Console (CLI)
# ------------------------------------------------------------
class ConsoleAgent(BaseAgent):
    def __init__(self, bus: MessageBus):
        super().__init__("Console", bus, None)
        self.current_question_id = None
        self.waiting_for_answer = False
        self.available_benefits = {}
        self._pending_question = None  # guarda a pergunta até receber benefícios
        self.all_benefits = {
            "Ajuda dos universitários": "Eles votam na alternativa que acreditam ser a correta",
            "Cartas": "Elimina 2 alternativas incorretas",
            "Pular": "Pular uma questão",
            "Mudar": "Mudar a pergunta para um tema escolhido"
        }

    def format_benefit_menu(self):
        """Formata o menu de benefícios mostrando todos, mas indicando quais estão disponíveis."""
        menu = ["\nBenefícios disponíveis:", "-" * 30, "0. Responder normalmente"]
        
        # Map benefit names to their menu index for consistent ordering
        benefit_order = {
            "Ajuda dos universitários": 1,
            "Cartas": 2,
            "Pular": 3,
            "Mudar": 4
        }
        
        # Create menu items
        menu_items = []
        for benefit_name, description in self.all_benefits.items():
            idx = benefit_order[benefit_name]
            qty = self.available_benefits.get(benefit_name, 0)
            status = f"(Quantidade: {qty})" if qty > 0 else "(Indisponível)"
            menu_items.append((idx, f"{idx}. {benefit_name} {status}"))
        
        # Add sorted menu items
        menu.extend(item[1] for item in sorted(menu_items))
        return menu

    def ask_for_answer(self):
        if self.current_question_id and self.waiting_for_answer:
            resp = input("\nSua resposta (A/B/C/D): ").strip().upper()
            if resp in ['A', 'B', 'C', 'D']:
                self.waiting_for_answer = False
                self.bus.send("Console", "Questions",
                              f"CHECK_ANSWER {self.current_question_id} {resp}")
            else:
                print("Por favor, digite A, B, C ou D")
                self.ask_for_answer()

    def receive(self, message: Message):
        # ----- chegou a pergunta: apenas armazena -----
        if message.content.startswith("QUESTION_DATA"):
            parts = message.content.split("|||")
            if len(parts) >= 3:
                header, q_text, alt_text = parts[0], parts[1], parts[2]
                qid = header.split()[1]

                self.current_question_id = qid
                self._pending_question = (qid, q_text, alt_text)
                self.bus.send("Console", "Benefits", "GET_AVAILABLE_BENEFITS")
            return

        # ----- chegaram os benefícios: mostra tudo -----
        if message.content.startswith("AVAILABLE_BENEFITS"):
            benefits_data = message.content.split(" ", 1)[1]
            self.available_benefits = eval(benefits_data)

            if self._pending_question:
                qid, q_text, alt_text = self._pending_question
                print(f"\nPergunta {qid}:")
                print("=" * 40)
                print(q_text)
                print("=" * 40)
                print(alt_text)
                self._pending_question = None

            # Display formatted benefit menu
            for line in self.format_benefit_menu():
                print(line)

            choice = input("\nEscolha um benefício (0-4): ").strip()
            if choice == "1" and "Ajuda dos universitários" in self.available_benefits:
                self.bus.send("Console", "Benefits", "USE_BENEFIT Ajuda dos universitários")
            elif choice == "2" and "Cartas" in self.available_benefits:
                self.bus.send("Console", "Benefits", "USE_BENEFIT Cartas")
            elif choice == "3" and "Pular" in self.available_benefits:
                self.bus.send("Console", "Benefits", "USE_BENEFIT Pular")
            elif choice == "4" and "Mudar" in self.available_benefits:
                self.bus.send("Console", "Benefits", "USE_BENEFIT Mudar")
            elif choice == "0":
                self.waiting_for_answer = True
                self.ask_for_answer()
            else:
                print("Opção inválida ou benefício não disponível!")
                self.waiting_for_answer = True
                self.ask_for_answer()
            return

        # ----- resto inalterado -----
        if message.content.startswith("ANSWER_RESULT"):
            _, qid, result, correct = message.content.split()
            print(f"\nResultado da pergunta {qid}: {result}")
            print(f"Resposta correta: {correct}")
            if result == "WRONG":
                print("\nGame Over! Você errou uma questão.")
                self.bus.send("Console", "Strategy", "GAME_OVER")

        elif message.content.startswith("BENEFIT_USED"):
            # Split message into parts
            parts = message.content.split("\n", 1)
            header = parts[0]
            effect = parts[1] if len(parts) > 1 else ""
            
            # Extract benefit name and description
            benefit_info = header.split(" ", 2)
            benefit_name = benefit_info[1]
            
            # Display benefit usage and effect
            print(f"\nBenefício '{benefit_name}' usado!")
            if effect:
                print(effect)
            
            # Ask for answer if not using Pular or Mudar
            if not benefit_name.startswith(("Pular", "Mudar")):
                self.waiting_for_answer = True
                self.ask_for_answer()

        elif message.content.startswith("BENEFIT_NOT_AVAILABLE"):
            print("\nEste benefício não está mais disponível!")
            self.waiting_for_answer = True
            self.ask_for_answer()

        else:
            print(f"[INFO] {message.content}")

# ------------------------------------------------------------
# Carregamento de CSVs
# ------------------------------------------------------------

def load_questions(csv_path: str) -> List[Question]:
    """Aceita dois formatos de perguntas."""
    df = pd.read_csv(csv_path)
    questions: List[Question] = []
    for i, r in df.iterrows():
        # Detecta formato
        if "Alternativas" in df.columns:
            alt_dict: Dict[str, str] = {}
            for alt in str(r["Alternativas"]).split(","):
                part = alt.strip()
                if ")" in part:
                    k, v = part.split(")", 1)
                    alt_dict[k.strip()] = v.strip()
            if len(alt_dict) != 4:
                parts = [p.strip() for p in str(r["Alternativas"]).split(",")]
                while len(parts) < 4:
                    parts.append("")
                alt_dict = {chr(65 + j): parts[j] for j in range(4)}
        else:
            alt_dict = {
                "A": str(r["AlternativaA"]),
                "B": str(r["AlternativaB"]),
                "C": str(r["AlternativaC"]),
                "D": str(r["AlternativaD"]),
            }

        questions.append(
            Question(
                id=int(r["ID"]) if "ID" in df.columns else i + 1,
                enunciado=str(r["Pergunta"]),
                alternativas=alt_dict,
                resposta=str(r["Resposta"]).strip(),
                dificuldade=int(r["Dificuldade"]),
                tema=str(r.get("Tema", "")),
            )
        )
    return questions


def load_benefits(csv_path: str) -> Dict[str, Benefit]:
    df = pd.read_csv(csv_path)
    return {
        row["Descrição"]: Benefit(row["Descrição"], int(row["Quantidade"]))
        for _, row in df.iterrows()
    }


def load_strategy(csv_path: str) -> List[Dict[str, any]]:
    df = pd.read_csv(csv_path)
    sequence = []
    for _, row in df.iterrows():
        difficulty = int(row["Dificuldade"])
        theme = str(row["Tema"]).strip() if pd.notna(row["Tema"]) else ""
        for part in str(row["Pergunta"]).split(','):
            if part.strip():
                sequence.append({
                    "id": int(part.strip()),
                    "difficulty": difficulty,
                    "theme": theme
                })
    return sequence

# ------------------------------------------------------------
# Função principal
# ------------------------------------------------------------

def main(questions_csv: str, benefits_csv: str, strategy_csv: str):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    llm = load_llm("/home/lucassaraiva5/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    bus = MessageBus()

    # Instancia agentes
    QuestionAgent(bus, llm, load_questions(questions_csv))
    BenefitAgent(bus, llm, load_benefits(benefits_csv))
    StrategyAgent(bus, llm, load_strategy(strategy_csv))
    ConsoleAgent(bus)

    # Inicia jogo
    bus.agents["Strategy"].start()

    # Exibe log ao final
    print("=== Log interno dos agentes ===")
    for m in bus.conversation_log():
        print(f"{m.sender} -> {m.receiver}: {m.content}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Quiz Multi‑Agente")
    parser.add_argument("--questions", default="banco_perguntas.csv")
    parser.add_argument("--benefits", default="beneficios.csv")
    parser.add_argument("--strategy", default="estrategia.csv")
    args = parser.parse_args()

    main(args.questions, args.benefits, args.strategy)
