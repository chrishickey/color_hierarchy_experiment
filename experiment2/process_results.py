import json
from collections import defaultdict

## WRITE YOUR OWN JSON FILE IF YOU HAVE RUN THE MODEL YOURSELF  AND WANT TO SEE !
JSON_FILE = 'results/results_mAP69.2.json'  # 69.2% mAP results used in workshop!

CUT_OFF = 1 # Exclude any results where less than 10% of pixels map to a single color
FOUND_KEY = 'found_{}'.format(CUT_OFF)
TOTAL_KEY = 'total_{}'.format(CUT_OFF)
with open(JSON_FILE) as fh:
    results = json.load(fh)
THRESHOLD = 50
MIN_NUM = 3
found = results[FOUND_KEY]
total = results[TOTAL_KEY]

colorz = ['Red', 'Green', 'Yellow', 'Blue', 'Brown', 'Pink', 'Purple', 'Orange', 'Gray']
adjs = ['Vivid', 'Strong', 'Deep', 'Light', 'Brilliant', 'Moderate', 'Dark', 'Pale']

color_keys = []
for col in colorz:
    for adj in adjs:
        color_keys.append('{}_{}'.format(adj, col))
        color_keys.append('Very_{}_{}'.format(adj, col))

stripped_total = {}
for item, color_dict in total.items():
    stripped_total[item] = defaultdict(int)
    for col, amount in color_dict.items():
        if col in color_keys:
            color_name = col.split('_')[-1]
            stripped_total[item][color_name] += amount

stripped_found = {}
for item, color_dict in found.items():
    stripped_found[item] = defaultdict(int)
    for col, amount in color_dict.items():
        if col in color_keys:
            color_name = col.split('_')[-1]
            stripped_found[item][color_name] += amount


recall_dict = {}
col_counter = defaultdict(int)
for item, total_dict in stripped_total.items():
    recall_dict[item] = defaultdict(int)
    found_dict = stripped_found[item]

    for col, total_amount in total_dict.items():
        if total_amount >= THRESHOLD:
            found_amount = found_dict[col]
            if len([k for k, v in total_dict.items() if v > THRESHOLD]) > MIN_NUM:
                col_counter[col] += total_amount
            if found_amount == 0:
                recall = 0
            else:
                recall = found_amount / total_amount
            recall_dict[item][col] = (recall, total_amount)

for col, amount in col_counter.items():
    print(col, amount)

recall_per_cat = defaultdict(list)

for item, color_recall_dict in recall_dict.items():
    sorted_dict = {k: v for k, v in
                   sorted(color_recall_dict.items(),
                    key=lambda item: item[1][0], reverse=True)}
    if len(sorted_dict) > MIN_NUM:
        print('\n', item, '\n')
        for key, val in sorted_dict.items():
            print(key, val[0], val[1])
            recall_per_cat[key].append(val[0])

print('\nTotal\n')
final_recall_dict = {}
for k, v in recall_per_cat.items():
    final_recall_dict[k] =  sum(v)/ len(v)

final_recall_dict = {k: v for k, v in
                sorted(final_recall_dict.items(),
                key=lambda item: item[1], reverse=True)}

for key, val in final_recall_dict.items():
    print(key, val)

print('\n')
print('Latex Table Colors!!')
keys = [k for k in final_recall_dict ]
print('Category', '&', ' & '.join(keys))
for obj, new_dict in recall_dict.items():
    if len(new_dict) > MIN_NUM:
        print(obj, '&', ' & '.join('-' if k not in new_dict else str(round(new_dict[k][0], 3 )).replace('0.', '.') for k in keys))
print('Mean', '&', ' & '.join([str(round(final_recall_dict[k], 3 )).replace('0.', '.') for k in final_recall_dict]))


################################ ADJECTIVE ################
stripped_total = {}
for item, color_dict in total.items():
    stripped_total[item] = defaultdict(int)
    for col, amount in color_dict.items():
        for adj in adjs:
            if adj == col[:len(adj)]:
                stripped_total[item][adj] += amount

stripped_found = {}
for item, color_dict in found.items():
    stripped_found[item] = defaultdict(int)
    for col, amount in color_dict.items():
        for adj in adjs:
            if adj == col[:len(adj)]:
                stripped_found[item][adj] += amount

recall_dict = {}
for item, total_dict in stripped_total.items():
    recall_dict[item] = defaultdict(int)
    found_dict = stripped_found[item]
    for col, total_amount in total_dict.items():
        if total_amount >= THRESHOLD:
            found_amount = found_dict[col]
            if found_amount == 0:
                recall = 0
            else:
                recall = found_amount / total_amount
            recall_dict[item][col] = (recall, total_amount)

recall_per_cat = defaultdict(list)

for item, color_recall_dict in recall_dict.items():
    sorted_dict = {k: v for k, v in
                   sorted(color_recall_dict.items(),
                    key=lambda item: item[1][0], reverse=True)}

    if len(sorted_dict) > MIN_NUM:
        print('\n', item, '\n')
        for key, val in sorted_dict.items():
            print(key, val[0], val[1])
            recall_per_cat[key].append(val[0])


print('\nTotal\n')
final_recall_dict = {}
for k, v in recall_per_cat.items():
    final_recall_dict[k] =  sum(v)/ len(v)

final_recall_dict = {k: v for k, v in
                sorted(final_recall_dict.items(),
                key=lambda item: item[1], reverse=True)}

for key, val in final_recall_dict.items():
    print(key, val)

print('\n')
keys = [k for k in final_recall_dict ]
print('Latex Table Adjectives!!')
print('Category', '&', ' & '.join(keys))
for obj, new_dict in recall_dict.items():
    if len(new_dict) > MIN_NUM:
        print(obj, '&', ' & '.join('-' if k not in new_dict else str(round(new_dict[k][0], 3 )).replace('0.', '.') for k in keys))
print('Mean', '&', ' & '.join([str(round(final_recall_dict[k], 3 )).replace('0.', '.') for k in final_recall_dict]))
