def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def test_merge_dicts():
    x = {'name':'Jack', 'age':'20'}
    y = {'sex':'man'}
    z = merge_dicts(x, y)
    assert z == {'name':'Jack', 'age':'20', 'sex':'man'}
