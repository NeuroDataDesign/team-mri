def flatten(current, result=[]):
    if isinstance(current, dict):
        for key in current:
            flatten(current[key], result)
    else:
        result.append(current)
    return result

def test_flatten():
    current = {'name':'liming', 'age':'20'}
    value = flatten(current, [])
    assert value == ['liming', '20']

