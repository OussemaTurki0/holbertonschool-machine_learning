def list_all(mongo_collection):
    """
    Returns all documents in the collection as a list.
    Returns empty list if no documents.
    """
    if mongo_collection is None:
        return []

    return list(mongo_collection.find())
