def get(*args, **kwargs):
    class Dummy:
        status_code = 404
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size=8192):
            return []
    return Dummy()

def post(*args, **kwargs):
    class Resp:
        status_code = 500
        text = ""
        def json(self):
            return {}
    return Resp()
