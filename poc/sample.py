def retrieve(query):
    """One function should stay one chunk."""
    return query.strip()


class Index:
    def add(self, chunk):
        self._items = list(self._items) + [chunk]

    def search(self, query):
        return retrieve(query)
