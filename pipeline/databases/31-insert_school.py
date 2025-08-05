def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document in the collection using kwargs.
    Returns the inserted document _id.
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
