'''去重测试'''
seen = set()
entitise_name_result = []

names = ['a', 'b', 'a', 'c', 'd', 'b']

for name in names:
    if name not in seen:
        seen.add(name)
        entitise_name_result.append(name)

print(entitise_name_result)
print(list(set(names)))


'''截断测试'''
def truncate_entity_name_length(entity_name: str) -> str:
    name = entity_name.strip()
    
    return name if len(name) < 15 else name[:15]

name = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
name2 = 'aaaa'

print(truncate_entity_name_length(name))
print(name2[:15])