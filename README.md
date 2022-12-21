# AI
**1.Write a program to implement breadth first search using python.<br>**
graph = {<br>
'1' : ['2','10'],<br>
'2' : ['3','8'],<br>
'3' : ['4'],<br>
'4' : ['5','6','7'],<br>
'5' : [],<br>
'6' : [],<br>
'7' : [],<br>
'8' : ['9'],<br>
'9' : [],<br>
'10' : []<br>
}<br>
visited = []<br>
queue = []<br>
def bfs(visited, graph, node):<br>
visited.append(node)<br>
queue.append(node)<br>
while queue:<br>
m = queue.pop(0)<br>
print (m, end = " ")<br>
for neighbour in graph[m]:<br>
if neighbour not in visited:<br>
visited.append(neighbour)<br>
queue.append(neighbour)<br>
print("Following is the Breadth-First Search")<br>
bfs(visited, graph, '1')<br>
Output:-<br>
Following is the Breadth-First Search<br>
1 2 10 3 8 4 9 5 6 7<br>
**2.write a program to implement a deapth first search using python**.
graph = {<br>
'5' : ['3','7'],<br>
'3' : ['2', '4'],<br>
'7' : ['6'],<br>
'6': [],<br>
'2' : ['1'],<br>
'1':[],<br>
'4' : ['8'],<br>
'8' : []<br>
}<br>
visited = set()<br>
def dfs(visited, graph, node):<br>
if node not in visited:<br>
print (node)<br>
visited.add(node)<br>
for neighbour in graph[node]:<br>
dfs(visited, graph, neighbour)<br>
print("Following is the Depth-First Search")<br>
dfs(visited, graph, '5')<br>
Output:-<br>
Following is the Depth-First Search<br>
5
3
2
1
4
8
7
6
<br>
**3.write a program to implement water jug problem using python.<br>**
from collections import defaultdict<br>
jug1, jug2, aim = 4, 3, 2<br>
visited = defaultdict(lambda: False)<br>
def waterJugSolver(amt1, amt2):<br>
if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):<br>
print(amt1, amt2)<br>
return True<br>
if visited[(amt1, amt2)] == False:<br>
print(amt1, amt2)<br>
visited[(amt1, amt2)] = True<br>
return (waterJugSolver(0, amt2) or<br>
waterJugSolver(amt1, 0) or<br>
waterJugSolver(jug1, amt2) or<br>
waterJugSolver(amt1, jug2) or<br>
waterJugSolver(amt1 + min(amt2, (jug1-amt1)),<br>
amt2 - min(amt2, (jug1-amt1))) or<br>
waterJugSolver(amt1 - min(amt1, (jug2-amt2)),<br>
amt2 + min(amt1, (jug2-amt2))))<br>
else:<br>
return False<br>
print("Steps: ")<br>
waterJugSolver(0, 0)<br>

Output:-<br>
Steps:
0 0
4 0
4 3
0 3
3 0
3 3
4 2
0 2
True<br>

**4.write a program to implement tower of hannoi using python.<br>**
def TowerOfHanoi(n , source, destination, auxiliary):<br>
if n==1:<br>
print ("Move disk 1 from source",source,"to destination",destination)<br>
return<br>
TowerOfHanoi(n-1, source, auxiliary, destination)<br>
print ("Move disk",n,"from source",source,"to destination",destination)<br>
TowerOfHanoi(n-1, auxiliary, destination, source)<br>
n = 4<br>
TowerOfHanoi(n,'A','B','C')<br>

Output:-
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 3 from source A to destination C<br>
Move disk 1 from source B to destination A<br>
Move disk 2 from source B to destination C<br>
Move disk 1 from source A to destination C<br>
Move disk 4 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 2 from source C to destination A<br>
Move disk 1 from source B to destination A<br>
Move disk 3 from source C to destination B<br>
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>

#5.Write a Program to Implement Best First Search using Python.<br>
from queue import PriorityQueue<br>
import matplotlib.pyplot as plt<br>
import networkx as nx<br>

 # for implementing BFS | returns path having lowest cost<br>
def best_first_search(source, target, n):<br>
    visited = [0] *n<br>
    visited[source] = True<br>
    pq = PriorityQueue()<br>
    pq.put((0, source))<br>
    while pq.empty() == False:<br>
        u = pq.get()[1]<br>
        print(u, end=" ") # the path having lowest cost<br>
        if u == target:<br>
            break<br>
        for v, c in graph[u]:<br>
            if visited[v] == False:<br>
                visited[v] = True<br>
                pq.put((c,v))
print()<br>

  # for adding edges to graph<br><br>
def addedge(x, y, cost):<br>
    graph[x].append((y, cost))<br>
    graph[y].append((x, cost))<br>
<br>
v = int(input("Enter the number of nodes: "))<br>
graph = [[] for i in range(v)] # undirected Graph<br>

e = int(input("Enter the number of edges: "))<br>
print("Enter the edges along with their weights:")<br>
for i in range(e):<br>
    x, y, z = list(map(int, input().split()))<br>
    addedge(x, y, z)<br>
source = int(input("Enter the Source Node: "))<br>
target = int(input("Enter the Target/Destination Node: "))<br>
print("\nPath: ", end = "")<br>
best_first_search(source, target, v)<br>
