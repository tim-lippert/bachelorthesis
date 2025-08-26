# Bachelorarbeit

Dieses Projekt stellt eine Implementierung von Sparse Einsum via mapping auf eine Sparse Batch Matrixmultiplikation dar. 

Es gibt zwei Versionen: sparse_einsum_v1 und sparse_einsum_v2, die sich im BMM-Kernel unterscheiden. In den meisten spärlichen Anwendungen sollte sparse_einsum_v1 das Ergebnis schneller berechnen. 

# Funktionsweise
Diese Funktionen nehmen einen SSA-Pfad und eine beliebige Anzahl an torch.sparse_coo_tensor Objekten entgegen und werten diese Berechnung aus. Ein SSA-Pfad definiert eine Abfolge an paarweisen Kontraktionen (tensorID1, tensorID2, subscript). Der Pfad [(subscript,0,1), (subscript,2,1)] kontrahiert beispielsweise bei zwei Eingabetensoren erst Tensor 1 mit Tensor 2, und kontrahiert im Anschluss diesen Ergebnistensor erneut mit Tensor 2. Für paarweise Transaktionen ist dieser trivial, für komplexere Kontraktionen kann dieser mit Bibliotheken wie [einsum_benchmark](https://benchmark.einsum.org/)  ermittelt werden.

Zur Nutzung dieser Funktion in eigenen Projekten müssen der ips4o- und src-Ordner sowie die Datei CMakeLists.txt und wrapper.py in den Projektordner kopiert werden. Anschließend kann das Modul mit den Befehlen cmake -S . -B build und anschließendem cmake --build build kompiliert werden. Im Anschluss können die Funktionen sparse_einsum_v1 bzw sparse_einsum_v2 aus wrapper.py importiert und genutzt werden.
