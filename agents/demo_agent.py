import json
import re
import time
from pathlib import Path

from agents.base_agent import AgentResult, BaseAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DemoAgent(BaseAgent):
    name = "demo"

    def __init__(self, catalog_path: str | Path | None = None):
        path = Path(catalog_path) if catalog_path else PROJECT_ROOT / "data" / "product_catalog.json"
        self.catalog = self._load_catalog(path)
        self._aliases = self._build_aliases()

    @staticmethod
    def _load_catalog(path: Path) -> dict[str, dict]:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return {item["name"]: item for item in payload["products"]}

    def _build_aliases(self) -> dict[str, str]:
        aliases = {}
        for name in self.catalog:
            aliases[re.sub(r"\s+", "", name.lower())] = name
        return aliases

    def run(self, user_input: str) -> AgentResult:
        start = time.perf_counter()
        text = str(user_input or "").strip()
        tool_calls: list[dict] = []
        metadata: dict = {"tool_results": {}}

        product = self._find_product(text)
        if product:
            answer = self._handle_product(text, product, tool_calls, metadata)
        elif self._is_greeting(text):
            answer = "你好，我是AgentEval Demo Agent，可以帮你查询产品价格、库存和总价。"
        elif self._is_help_query(text):
            answer = "我可以帮你查询产品价格、库存和总价。"
        elif self._is_product_list_query(text):
            answer = "我们目前在售：" + "、".join(self.catalog) + "。"
        elif self._is_out_of_scope(text):
            answer = "抱歉，我无法回答这个问题，我只能回答产品价格、库存和购买相关的问题。"
        elif self._is_product_related(text):
            answer = "你想查询哪款耳机？请告诉我具体型号。"
        else:
            answer = "我没理解你的意思，请告诉我你想查询的产品型号。"

        latency = time.perf_counter() - start
        return AgentResult(
            answer=answer,
            tool_calls=tool_calls,
            trajectory=[call["name"] for call in tool_calls],
            latency=round(latency, 4),
            metadata=metadata,
        )

    def _handle_product(self, text: str, product: str, tool_calls: list[dict], metadata: dict) -> str:
        if self._wants_inventory(text):
            search = self._call_tool(tool_calls, metadata, "search_product", {"product": product})
            if not search.get("ok"):
                return self._missing_answer(product)
            inventory = self._call_tool(tool_calls, metadata, "check_inventory", {"product": product})
            return self._inventory_answer(product, inventory)

        if self._wants_total(text):
            quantity = self._parse_quantity(text)
            if quantity is None:
                return f"请告诉我要购买多少个{product}。"
            search = self._call_tool(tool_calls, metadata, "search_product", {"product": product})
            if not search.get("ok"):
                return self._missing_answer(product)
            calc = self._call_tool(tool_calls, metadata, "calculate_price", {"product": product, "quantity": quantity})
            return self._total_answer(product, calc)

        search = self._call_tool(tool_calls, metadata, "search_product", {"product": product})
        if not search.get("ok"):
            return self._missing_answer(product)
        return self._price_answer(product, search)

    def _call_tool(self, tool_calls: list[dict], metadata: dict, name: str, arguments: dict) -> dict:
        result = self._run_tool(name, arguments)
        tool_calls.append({"name": name, "arguments": arguments})
        metadata["tool_results"].setdefault(name, []).append(result)
        return result

    def _run_tool(self, name: str, arguments: dict) -> dict:
        if name == "search_product":
            return self._search_product(arguments.get("product", ""))
        if name == "check_inventory":
            return self._check_inventory(arguments.get("product", ""))
        if name == "calculate_price":
            return self._calculate_price(arguments.get("product", ""), int(arguments.get("quantity", 0) or 0))
        return {"ok": False, "error": f"Unknown tool: {name}"}

    def _search_product(self, product: str) -> dict:
        record = self.catalog.get(product)
        if not record:
            return {"ok": False, "product": product, "error": "Product not found"}
        return {"ok": True, "product": product, "price": record["price"], "stock": record["stock"]}

    def _check_inventory(self, product: str) -> dict:
        record = self.catalog.get(product)
        if not record:
            return {"ok": False, "product": product, "error": "Product not found"}
        return {"ok": True, "product": product, "available": record["stock"] > 0, "stock": record["stock"]}

    def _calculate_price(self, product: str, quantity: int) -> dict:
        record = self.catalog.get(product)
        if not record:
            return {"ok": False, "product": product, "error": "Product not found"}
        return {
            "ok": True,
            "product": product,
            "quantity": quantity,
            "unit_price": record["price"],
            "total": record["price"] * quantity,
        }

    @staticmethod
    def _price_answer(product: str, result: dict) -> str:
        return f"{product}售价为{result['price']}元"

    @staticmethod
    def _inventory_answer(product: str, result: dict) -> str:
        if result.get("available"):
            return f"{product}当前有货，剩余{result['stock']}件"
        return f"{product}当前缺货"

    @staticmethod
    def _total_answer(product: str, result: dict) -> str:
        return f"{product}总价为{result['total']}元"

    @staticmethod
    def _missing_answer(product: str) -> str:
        return f"未找到产品 {product}"

    def _find_product(self, text: str) -> str | None:
        compact = re.sub(r"\s+", "", text.lower())
        for alias in sorted(self._aliases, key=len, reverse=True):
            if alias in compact:
                return self._aliases[alias]
        brand_match = re.search(r"(openfit\s*\d+|openfit\s+(?:air|pro)|comfobuds\s+mini)", text, re.IGNORECASE)
        if brand_match:
            return brand_match.group(1).strip().title()
        return None

    @staticmethod
    def _wants_inventory(text: str) -> bool:
        return any(k in text for k in ("库存", "有货", "现货", "缺货"))

    @staticmethod
    def _wants_total(text: str) -> bool:
        if any(k in text for k in ("总价", "总金额")):
            return True
        return bool(re.search(r"\d+\s*个", text)) and "多少钱" in text

    @staticmethod
    def _parse_quantity(text: str) -> int | None:
        match = re.search(r"(\d+)\s*个", text)
        return int(match.group(1)) if match else None

    @staticmethod
    def _is_greeting(text: str) -> bool:
        lower = text.lower()
        return any(lower.startswith(k) for k in ("你好", "hello", "hi", "嗨"))

    @staticmethod
    def _is_help_query(text: str) -> bool:
        return any(k in text for k in ("帮助", "做什么", "能做什么"))

    @staticmethod
    def _is_product_list_query(text: str) -> bool:
        return any(k in text for k in ("卖什么", "有哪些", "在售"))

    @staticmethod
    def _is_out_of_scope(text: str) -> bool:
        return any(k in text for k in ("天气", "weather", "股票", "电影"))

    @staticmethod
    def _is_product_related(text: str) -> bool:
        return any(k in text for k in ("耳机", "价格", "多少钱", "库存", "有货", "产品"))
