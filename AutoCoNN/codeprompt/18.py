def ralign(sop, default='-'):
    c = rasp.Aggregate(rasp.Select(sop, rasp.Map(lambda x: '_', sop), rasp.Comparison.EQ), rasp.Map(lambda x: 1, sop))
    return rasp.Aggregate(rasp.Select(rasp.indices + c, rasp.indices, rasp.Comparison.EQ), sop, default='-')


