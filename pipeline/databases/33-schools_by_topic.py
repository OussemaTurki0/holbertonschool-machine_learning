def schools_by_topic(mongo_collection, topic):
    """
    Returns list of schools that have 'topic' in their topics list.
    """
    return list(mongo_collection.find({"topics": topic}))
