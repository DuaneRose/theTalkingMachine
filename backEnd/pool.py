query = ""

def get_query():
    global query
    return query

def set_query(new_query):
    global query
    query = new_query
    

def delete_query():
    global query
    query = ""