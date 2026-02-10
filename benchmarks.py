import subprocess

targets = [
    'examples/reset.txt',
    'examples/circular.txt',
    'examples/mutex1.txt',
    'examples/mutex2.txt',
    'examples/spinlock.txt'
]

state_domains = [
    'constants',
    'disjunctive constants'
]

def main():
    out = 'results.txt'
    file = open(out, 'w')
    file.write('')
    file.close()
    for target in targets:
        for domain in state_domains:
            subprocess.run(['python', 'main.py', target, domain, '-o', out])
            subprocess.run(['python', 'main.py', target, domain, '-o', out, '-t'])

if __name__ == '__main__':
    main()