c = open('src/ml/auto_maintenance.py', 'r', encoding='utf-8').read()
old = '''if __name__ == '__main__':
    """Run maintenance jobs manually."""
    import argparse'''
new = '''if __name__ == '__main__':
    """Run maintenance jobs manually."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import argparse'''
c = c.replace(old, new)
open('src/ml/auto_maintenance.py', 'w', encoding='utf-8').write(c)
print('Fixed logging in __main__')
