def update_topics(mongo_collection, name, topics):
    """
    Update all documents matching 'name' by setting 'topics' field.
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
