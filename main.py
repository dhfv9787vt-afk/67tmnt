"""Your first PyEditor project."""

import json
import random
from typing import Dict, Any, Tuple, List

class FinancialAuditEnv:
    """
    Симулятор финансовой среды для ИИ-аудитора (Gymnasium-style контракт).
    Изолирует закрытый эталон (truth.json) от агента и оценивает его выводы.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {"max_steps": 100}
        self.max_steps = self.config.get("max_steps", 100)
        self.current_step = 0
        self.episode_id = "ep_hackathon_001"
        
        # Состояние эпизода
        self.manifest = {}
        self.ledger = []
        self.truth = []  # Закрытый эталон, агент к нему доступа не имеет
        self.terminated = False
        self.truncated = False

    def reset(
        self, 
        seed: int = None, 
        config: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Сбрасывает среду и генерирует новый эпизод с заданным seed."""
        if seed is not None:
            random.seed(seed)
            
        self.current_step = 0
        self.terminated = False
        self.truncated = False
        
        # 1. Формируем манифест эпизода
        self.manifest = {
            "schema_version": "1.0",
            "episode_id": self.episode_id,
            "base_currency": "USD",
            "period_start": "2026-01-01T00:00:00Z",
            "period_end": "2026-01-31T23:59:59Z",
            "timezone": "UTC",
            "total_records": 4,
            "rounding": "round_half_even"
        }
        
        # 2. Генерируем проводки (ledger.csv) с переносом строк для читаемости
        self.ledger = [
            {
                "row_id": "r1", 
                "timestamp": "2026-01-05T10:00:00Z", 
                "account": "1010", 
                "debit_minor": 15000, 
                "credit_minor": 0, 
                "currency": "USD", 
                "invoice_id": "inv_101"
            },
            {
                "row_id": "r2", 
                "timestamp": "2026-01-06T12:00:00Z", 
                "account": "2010", 
                "debit_minor": 0, 
                "credit_minor": 15000, 
                "currency": "USD", 
                "invoice_id": "inv_101"
            },
            # Ошибка 1: Дубль счета (r3 дублирует r1)
            {
                "row_id": "r3", 
                "timestamp": "2026-01-07T10:00:00Z", 
                "account": "1010", 
                "debit_minor": 15000, 
                "credit_minor": 0, 
                "currency": "USD", 
                "invoice_id": "inv_101"
            },
            {
                "row_id": "r4", 
                "timestamp": "2026-01-10T14:30:00Z", 
                "account": "3010", 
                "debit_minor": 5000, 
                "credit_minor": 0, 
                "currency": "USD", 
                "invoice_id": "inv_102"
            }
        ]
        
        # 3. Формируем скрытый эталон (truth.json)
        self.truth = [
            {
                "issue_id": "iss_01",
                "type": "duplicate_invoice",
                "affected_ids": ["r1", "r3"],
                "expected_amount_minor": 15000,
                "currency": "USD",
                "allowed_evidence_sets": [["r1", "r3"]]
            }
        ]
        
        # Первое наблюдение для агента (строго без truth.json!)
        observation = {
            "task": "Audit the ledger and submit findings using tools.",
            "manifest": self.manifest,
            "remaining_budget_steps": self.max_steps
        }
        
        info = {"info": "Episode initialized successfully."}
        return observation, info

    def step(
        self, 
        action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Обрабатывает действие агента и возвращает результаты шага."""
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is already finished. Call reset() first.")
            
        self.current_step += 1
        
        # Проверка лимита шагов
        if self.current_step > self.max_steps:
            self.truncated = True
            return {"error": "Step limit reached"}, 0.0, self.terminated, self.truncated, {"remaining_steps": 0}

        action_type = action.get("type")
        obs = {}
        reward = 0.0

        # --- Реализация инструментов из ТЗ (Раздел 04) ---
        if action_type == "list_records":
            cursor = action.get("cursor", 0)
            limit = min(action.get("limit", 50), 50)  # Максимум 50 строк за раз
            slice_records = self.ledger[cursor:cursor + limit]
            obs = {
                "table": "ledger",
                "records": slice_records,
                "next_cursor": cursor + limit if cursor + limit < len(self.ledger) else None
            }
            
        elif action_type == "get_record":
            row_id = action.get("id")
            found = next((r for r in self.ledger if r["row_id"] == row_id), None)
            if found:
                obs = {"record": found}
            else:
                obs = {"error": f"Record with id '{row_id}' not found"}
                
        elif action_type == "calculator":
            expr = action.get("expression", "")
            # Строго безопасный калькулятор (только цифры и базовая арифметика)
            allowed_chars = set("0123456789+-*/.() ")
            if all(c in allowed_chars for c in expr) and len(expr) <= 100:
                try:
                    result = eval(expr)
                    obs = {"result": result}
                except Exception:
                    obs = {"error": "Calculation evaluation error"}
            else:
                obs = {"error": "Invalid or unsafe characters in expression"}
                
        elif action_type == "submit":
            self.terminated = True
            findings = action.get("findings", [])
            
            # Проверка структуры выводов (максимум 30 выводов)
            if not isinstance(findings, list) or len(findings) > 30:
                reward = -1.0
                obs = {
                    "status": "error", 
                    "message": "Invalid findings structure or limit exceeded"
                }
            else:
                reward = self._compute_reward(findings)
                obs = {"status": "submitted", "reward": reward}
        else:
            obs = {"error": f"Unknown action type '{action_type}'"}

        info = {"remaining_steps": self.max_steps - self.current_step}
        return obs, reward, self.terminated, self.truncated, info

    def _compute_reward(self, findings: List[Dict[str, Any]]) -> float:
        """Считает Precision, Recall, F1 и терминальную награду по ТЗ (Раздел 05)."""
        tp = 0
        fp = 0
        fn = len(self.truth)
        
        for f in findings:
            claimed_type = f.get("type")
            affected_ids = set(f.get("affected_ids", []))
            
            # Проверяем совпадение с эталоном
            matched = False
            for t in self.truth:
                if claimed_type == t["type"] and affected_ids == set(t["affected_ids"]):
                    matched = True
                    break
            
            if matched:
                tp += 1
                fn -= 1
            else:
                fp += 1
                
        # Расчет F1-меры
        denominator = (2 * tp + fp + fn)
        if denominator == 0:
            f1 = 1.0 if len(self.truth) == 0 else 0.0
        else:
            f1 = (2 * tp) / denominator
            
        # Награда по формуле из задания: clip(2*F1 - 1 - 0.002 * steps, -1, 1)
        step_penalty = 0.002 * self.current_step
        raw_reward = 2 * f1 - 1 - step_penalty
        
        return max(-1.0, min(1.0, raw_reward))


# --- Демонстрация работы симулятора и скриптового агента ---
if __name__ == "__main__":
    print("=== Инициализация среды аудита ===")
    env = FinancialAuditEnv()
    obs, info = env.reset(seed=42)
    print("Начальное наблюдение агента:\n", json.dumps(obs, indent=2))
    
    print("\n=== Шаг 1: Агент запрашивает список записей ===")
    obs, reward, term, trunc, info = env.step({"type": "list_records", "limit": 10})
    print("Ответ среды:\n", json.dumps(obs, indent=2))
    
    print("\n=== Шаг 2: Агент использует калькулятор для проверки сумм ===")
    obs, reward, term, trunc, info = env.step({"type": "calculator", "expression": "15000 + 15000"})
    print("Результат вычислений:\n", json.dumps(obs, indent=2))
    
    print("\n=== Шаг 3: Агент отправляет найденную ошибку (submit) ===")
    submission = {
        "type": "submit",
        "findings": [
            {
                "type": "duplicate_invoice",
                "affected_ids": ["r1", "r3"],
                "claimed_amount_minor": 15000,
                "currency": "USD",
                "evidence_ids": ["r1", "r3"],
                "explanation": "Transactions r1 and r3 have identical amounts, accounts and invoice IDs."
            }
        ]
    }
    obs, reward, term, trunc, info = env.step(submission)
    print("Финальный ответ среды:\n", json.dumps(obs, indent=2))
    print(f"Итоговая награда агента: {reward:.4f}")