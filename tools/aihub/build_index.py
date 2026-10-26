"""라벨 zip 풀린 폴더 → label_index.json (대문자 이미지 stem → json 경로).  python tools/aihub/build_index.py <aihub_root>"""
import glob, os, json, sys
root = sys.argv[1] if len(sys.argv) > 1 else '.'
idx = {os.path.basename(f)[:-5].upper(): os.path.relpath(f, root) for f in glob.glob(os.path.join(root, 'labels', '*', '**', '*.json'), recursive=True)}
json.dump(idx, open(os.path.join(root, 'label_index.json'), 'w'))
print(len(idx))
