import pandas as pd
import logging
from dataclasses import dataclass
from typing import List, Dict

# ------------------------------------------------------------
# Dependência opcional para rodar Llama‑3 localmente
# ------------------------------------------------------------
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # fallback mock

DEFAULT_MODEL_PATH = "models/llama-3-8b-instruct.Q4_K_M.gguf"  # pode ser alterado via --model
MODEL_CTX = 4096
TOTAL_ROUNDS = 20  # número máximo de perguntas por partida

# ------------------------------------------------------------
# Utilitário LLM
# ------------------------------------------------------------

def load_llm(model_path: str):
    """Carrega o modelo se existir; caso contrário, devolve None (modo stub)."""
    if Llama is None:
        logging.warning("llama-cpp-python não instalado — respostas serão simuladas.")
        return None

    if not Path(model_path).is_file():
        logging.warning(f"Modelo não encontrado em '{model_path}'. Rodando em modo stub.")
        return None

    return Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=MODEL_CTX)

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

    def receive(self, message: Message):
        if message.content.startswith("GET_QUESTION"):
            _, qid = message.content.split()
            qid = int(qid)
            q = self.questions.get(qid)
            if not q:
                self.bus.send(self.name, message.sender, f"QUESTION_NOT_FOUND {qid}")
                return
            # envia somente ao solicitante (Strategy) - Strategy repassa ao Console
            self.bus.send(self.name, message.sender,
                           f"QUESTION_DATA {qid} {q.enunciado} | {q.alternativas}")

        elif message.content.startswith("CHECK_ANSWER"):
            _, qid, guess = message.content.split()
            qid = int(qid)
            q = self.questions.get(qid)
            if not q:
                self.bus.send(self.name, message.sender, f"QUESTION_NOT_FOUND {qid}")
                return
            result_str = "CORRECT" if guess.upper() == q.resposta.upper() else "WRONG"
            # envia resultado ao Console (feedback) e ao Strategy (fluxo do jogo)
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

    def receive(self, message: Message):
        if message.content == "GET_AVAILABLE_BENEFITS":
            data = {k: v.quantidade for k, v in self.benefits.items() if v.quantidade > 0}
            self.bus.send(self.name, message.sender, f"AVAILABLE_BENEFITS {data}")

        elif message.content.startswith("USE_BENEFIT"):
            _, key = message.content.split(None, 1)
            b = self.benefits.get(key)
            if b and b.quantidade > 0:
                b.quantidade -= 1
                self.bus.send(self.name, message.sender, f"BENEFIT_USED {key} {b.quantidade}")
            else:
                self.bus.send(self.name, message.sender, f"BENEFIT_NOT_AVAILABLE {key}")

# ------------------------------------------------------------
# Agente de Estratégia (ordem definida por CSV)
# ------------------------------------------------------------
class StrategyAgent(BaseAgent):
    def __init__(self, bus: MessageBus, llm, sequence: List[int]):
        super().__init__("Strategy", bus, llm)
        self.sequence = sequence
        self.idx = 0
        self.score = 0
        self.used_benefits: List[str] = []

    def start(self):
        if not self.sequence:
            self.bus.send(self.name, "Console", "GAME_OVER Sequência vazia")
            return
        self.bus.send(self.name, "Questions", f"GET_QUESTION {self.sequence[0]}")

    def receive(self, message: Message):
        if message.content.startswith("QUESTION_DATA"):
            # repassa apenas para o Console
            self.bus.send(self.name, "Console", message.content)

        elif message.content.startswith("ANSWER_RESULT"):
            _, qid, result, correct = message.content.split()
            if result == "CORRECT":
                self.score += 1
            else:
                if "Cartas" not in self.used_benefits:
                    self.used_benefits.append("Cartas")
                    self.bus.send(self.name, "Benefits", "USE_BENEFIT Cartas (Elimina 2 alternativas incorretas)")

            # próxima pergunta
            self.idx += 1
            if self.idx < len(self.sequence):
                self.bus.send(self.name, "Questions", f"GET_QUESTION {self.sequence[self.idx]}")
            else:
                self.bus.send(self.name, "Console", f"GAME_OVER Score: {self.score}/{len(self.sequence)}")

# ------------------------------------------------------------
# Agente Console (CLI)
# ------------------------------------------------------------
class ConsoleAgent(BaseAgent):
    def __init__(self, bus: MessageBus):
        super().__init__("Console", bus, None)

    def receive(self, message: Message):
        if message.content.startswith("QUESTION_DATA"):
            _, qid, texto, alts = message.content.split(" ", 3)
            print(f"\nPergunta {qid}: {texto}")
            print(alts)
            resp = input("Sua resposta (A/B/C/D): ")
            self.bus.send("Console", "Questions", f"CHECK_ANSWER {qid} {resp.strip()}")
        elif message.content.startswith("ANSWER_RESULT"):
            _, qid, result, correct = message.content.split()
            print(f"Resultado da pergunta {qid}: {result}. Resposta correta: {correct}")
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
    return {row["Descrição"]: Benefit(row["Descrição"], int(row["Quantidade"])) for _, row in df.iterrows()}


def load_strategy(csv_path: str) -> List[int]:
    df = pd.read_csv(csv_path)
    sequence: List[int] = []
    for _, row in df.iterrows():
        for part in str(row["Pergunta"]).split(','):
            if part.strip():
                sequence.append(int(part.strip()))
    return sequence

# ------------------------------------------------------------
# Função principal
# ------------------------------------------------------------

def main(questions_csv: str, benefits_csv: str, strategy_csv: str):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    llm = load_llm("/home/lucassaraiva5/models/llama-3-8b-instruct.Q4_K_M.gguf")
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
