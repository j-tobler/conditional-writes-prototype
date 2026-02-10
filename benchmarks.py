# Copyright 2026 The Commonwealth of Australia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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