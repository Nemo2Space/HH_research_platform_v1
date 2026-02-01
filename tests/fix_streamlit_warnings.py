import os
import glob

# Find all Python files in dashboard
files = glob.glob('dashboard/**/*.py', recursive=True)
files.extend(glob.glob('dashboard/*.py'))

total_replaced = 0

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Replace use_container_width=True with width='stretch'
        content = content.replace('use_container_width=True', "width='stretch'")
        # Replace use_container_width=False with width='content'  
        content = content.replace('use_container_width=False', "width='content'")
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            count = original.count('use_container_width')
            print(f"✅ {filepath}: replaced {count} occurrences")
            total_replaced += count
    except Exception as e:
        print(f"Error in {filepath}: {e}")

print(f"\nTotal replaced: {total_replaced}")
