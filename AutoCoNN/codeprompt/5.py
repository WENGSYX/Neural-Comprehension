def make_shuffle_dyck2() -> rasp.SOp:
    return make_shuffle_dyck(pairs=["()", "{}"]).named("shuffle_dyck2")


