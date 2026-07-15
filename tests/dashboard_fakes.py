"""In-memory fakes for Dashboard/ tests -- Redis ve MinIO (Dashboard/codes/minio_archiver.py
tarafinda kullanilan CIPLAK minio.Minio yuzeyi, src/common/fakes.py'deki
FakeMinioClient'tan FARKLI bir arayuz, o yuzden burada AYRI/kucuk bir fake var).

Bu modul kasitli olarak "test_" ile BASLAMIYOR -- pytest onu bir test dosyasi
olarak toplamiyor, sadece diger test dosyalarinin import ettigi bir yardimci.
"""

from __future__ import annotations

from contextlib import contextmanager


def find_by_id(node, target_id: str):
    """app_dash.layout (dash bilesen agaci) icinde verilen id'ye sahip
    ilk dugumu DFS ile bulur. Layout GERCEKTEN olustugu Python nesnesi
    oldugu icin (statik bir HTML dizesi/regex degil), bu, dogru
    id/sira/nesting'i dogrulamanin en guvenilir yolu -- kaynak metnini
    regex'le taramaktan cok daha az kirilgan."""
    if getattr(node, "id", None) == target_id:
        return node
    children = getattr(node, "children", None)
    if children is None:
        return None
    if not isinstance(children, list):
        children = [children]
    for child in children:
        if hasattr(child, "id") or hasattr(child, "children"):
            found = find_by_id(child, target_id)
            if found is not None:
                return found
    return None


@contextmanager
def simulate_trigger(prop_id: str | None, value=1):
    """dash.callback_context.triggered'i BIR callback disinda cagirinca
    dash.exceptions.MissingCallbackContextException firlatiyor -- Dash bunu
    normalde HTTP istek dispatch'i sirasinda bir contextvar'a yazip
    dolduruyor (bkz. dash/_callback_context.py). Bu, "ctx.callback_context"
    kullanan callback fonksiyonlarini (orn. update_map_style_setting,
    toggle_replay_open) DOGRUDAN cagirarak test edebilmek icin AYNI
    contextvar'i taklit eder -- gercek bir Dash sunucusu/istegi olmadan.

    prop_id=None -- "hic tetikleyici yok" durumunu (orn. Dash'in ilk
    yuklemede yaptigi gibi) simule eder -- ctx.triggered BOS (falsy) doner,
    MissingCallbackContextException FIRLAMAZ (context yine de "var",
    sadece icinde tetikleyici yok -- context'in HIC KURULMAMASINDAN
    (gercek Dash'te asla olmayan bir durum) farkli)."""
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    triggered_inputs = [] if prop_id is None else [{"prop_id": prop_id, "value": value}]
    token = context_value.set(AttributeDict(triggered_inputs=triggered_inputs))
    try:
        yield
    finally:
        context_value.reset(token)


class FakePipeline:
    """redis.Redis.pipeline()'in yerini tutar -- delete/srem cagrilarini
    SIRAYLA biriktirir, execute() cagrilana kadar hicbir sey uygulanmaz
    (gercek Redis pipeline'in atomik/toplu davranisiyla tutarli)."""

    def __init__(self, store: "FakeRedis") -> None:
        self._store = store
        self._ops: list[tuple[str, tuple]] = []

    def delete(self, key: str) -> "FakePipeline":
        self._ops.append(("delete", (key,)))
        return self

    def srem(self, key: str, *members: str) -> "FakePipeline":
        self._ops.append(("srem", (key, members)))
        return self

    def execute(self) -> list:
        results = []
        for op, args in self._ops:
            if op == "delete":
                (key,) = args
                results.append(self._store._data.pop(key, None) is not None)
            elif op == "srem":
                key, members = args
                s = self._store._sets.get(key, set())
                removed = len(s & set(members))
                s -= set(members)
                results.append(removed)
        self._ops = []
        return results


class FakeRedis:
    """dashboard_consumer.py (sweep_stale_flights) ve app.py'nin (_get_flights,
    _reverse_geocode) kullandigi Redis yuzeyinin kucuk, bellek-ici bir
    alt kumesi -- sadece testlerde kullanilan cagrilar implemente edildi."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}

    # -- string key/value --
    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._data[key] = value  # TTL (ex) testler icin onemsiz, gormezden geliniyor

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._data[key] = value  # TTL testler icin onemsiz, gormezden geliniyor

    def mget(self, keys: list[str]) -> list:
        return [self._data.get(k) for k in keys]

    # -- kume (set) --
    def sadd(self, key: str, *members: str) -> int:
        s = self._sets.setdefault(key, set())
        before = len(s)
        s.update(members)
        return len(s) - before

    def srem(self, key: str, *members: str) -> int:
        s = self._sets.get(key, set())
        removed = len(s & set(members))
        s -= set(members)
        return removed

    def smembers(self, key: str) -> set[str]:
        return set(self._sets.get(key, set()))

    # -- toplu silme --
    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)


class FakeResponse:
    """requests.get(...)'in dondurdugu Response nesnesinin yerini tutar --
    app.py cogunlukla sadece .json() cagiriyor, ama _reverse_geocode gibi
    bazi yerler ONCE .status_code'u kontrol ediyor (bkz. o fonksiyon) --
    varsayilan 200, gerekirse override edilebilir."""

    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeRequestsRouter:
    """dashapp.requests.get'in yerini tutar -- URL'deki bir alt dizeye
    (orn. "flights", "alerts", "replay_frame") gore FARKLI sahte yanit
    dondurur. side_effect olarak da bir fonksiyon verilebilir -- YAVAS/
    CAKISAN istekleri (sequence-guard/yaris durumu testleri) simule etmek
    icin cagrilir, gercek requests.get GIBI (url, ...) ile cagrilir."""

    def __init__(self, routes: dict[str, object], on_call=None) -> None:
        self._routes = routes  # {url_substring: payload}
        self._on_call = on_call  # opsiyonel: her cagrida calisir (yan etki icin)
        self.calls: list[str] = []

    def __call__(self, url, *args, **kwargs):
        self.calls.append(url)
        if self._on_call is not None:
            self._on_call(url)
        for substring, payload in self._routes.items():
            if substring in url:
                return FakeResponse(payload)
        return FakeResponse([])


class FakeMinio:
    """minio.Minio'nun Dashboard/codes/minio_archiver.py'nin kullandigi ciplak
    yuzeyi -- put_object/delete_bucket_lifecycle/bucket_exists/make_bucket."""

    def __init__(self) -> None:
        self.buckets: dict[str, dict[str, bytes]] = {}
        self.put_calls: list[dict] = []  # {bucket, object_name, content_type, data}
        self.lifecycle_delete_calls: list[str] = []
        self.raise_on_delete_lifecycle: Exception | None = None

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        self.buckets.setdefault(bucket_name, {})

    def put_object(self, bucket_name, object_name, data, length,
                   content_type="application/octet-stream", **_kwargs):
        payload = data.read() if hasattr(data, "read") else bytes(data)
        self.buckets.setdefault(bucket_name, {})[object_name] = payload
        self.put_calls.append({
            "bucket": bucket_name, "object_name": object_name,
            "content_type": content_type, "data": payload,
        })

    def delete_bucket_lifecycle(self, bucket_name: str) -> None:
        self.lifecycle_delete_calls.append(bucket_name)
        if self.raise_on_delete_lifecycle is not None:
            raise self.raise_on_delete_lifecycle
