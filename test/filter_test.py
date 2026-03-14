validated_item_names = ['a', 'b', 'c']


compact = ', '.join(f'"{item_name}"' for item_name in validated_item_names)
print(f'item_name in [{compact}]')


compact1 = f'{validated_item_names}'
print(f'item_name in {compact1}')


print(f'item_name in {validated_item_names}')