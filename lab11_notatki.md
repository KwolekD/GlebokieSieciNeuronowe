# Lista 11 — Notatki: minLSTM

**Modyfikacja:** Zamiana `nn.LSTM` (lista 9) na ręcznie zaimplementowaną komórkę minLSTM  
**Źródło:** Feng et al., *Were RNNs All We Needed?*, arXiv:2410.01201 (2024)  
**Datasety:** IMDB (analiza sentymentu), InsectWingbeat (klasyfikacja owadów)

---

## 1. Motywacja

Klasyczny LSTM ma dwa fundamentalne ograniczenia:

1. **Sekwencyjne BPTT** — każdy krok czasowy $t$ wymaga wyniku kroku $t-1$, więc obliczeń nie można zrównoleglić wzdłuż wymiaru czasu podczas uczenia. Dla sekwencji długości $T$ to $T$ sekwencyjnych operacji na GPU.

2. **Duża liczba parametrów** — $O(4 d_h(d_x + d_h))$, gdzie $d_h$ to rozmiar stanu ukrytego, $d_x$ to rozmiar wejścia. Przy $d_h = 2d_x$ jest to ~12 razy więcej niż minimalna liczba potrzebna do zdefiniowania rekurencji.

Oba problemy wynikają z tego samego źródła: bramki $f_t$, $i_t$, $o_t$ zależą od poprzedniego stanu ukrytego $h_{t-1}$, przez co wejścia do algorytmu zrównoleglającego nie są znane z góry.

---

## 2. Klasyczny LSTM — przypomnienie

$$f_t = \sigma(\text{Linear}([x_t, h_{t-1}]))$$
$$i_t = \sigma(\text{Linear}([x_t, h_{t-1}]))$$
$$o_t = \sigma(\text{Linear}([x_t, h_{t-1}]))$$
$$\tilde{c}_t = \tanh(\text{Linear}([x_t, h_{t-1}]))$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

Zależność od $h_{t-1}$ we wszystkich bramkach uniemożliwia zrównoleglenie.

---

## 3. Trzy kroki uproszczenia → minLSTM

### Krok 1: Usunięcie zależności od $h_{t-1}$ z bramek

Bramki przestają przyjmować $h_{t-1}$ jako wejście — zależą wyłącznie od $x_t$:

$$f_t = \sigma(\text{Linear}(x_t))$$
$$i_t = \sigma(\text{Linear}(x_t))$$
$$\tilde{c}_t = \tanh(\text{Linear}(x_t))$$

Teraz dla całej sekwencji $x_{1:T}$ wszystkie $f_t$, $i_t$, $\tilde{c}_t$ można policzyć **jednocześnie**, zanim w ogóle zacznie się rekurencja.

### Krok 2: Usunięcie `tanh` z kandydującego stanu

`tanh` w oryginalnym LSTM służył do stabilizacji gradientów przy mnożeniu przez $h_{t-1}$. Po kroku 1 ta zależność zniknęła, więc `tanh` jest zbędny:

$$\tilde{h}_t = \text{Linear}(x_t)$$

### Krok 3: Normalizacja bramek (time-independence)

W LSTM bramki $f_t$ i $i_t$ są niezależne — mogą obie być bliskie 1, co powoduje że $c_t$ rośnie z długością sekwencji. Utrudnia to optymalizację gdy target jest niezależny od długości sekwencji (np. klasyfikacja).

Rozwiązanie: wymuś $f'_t + i'_t = 1$:

$$f'_t = \frac{f_t}{f_t + i_t}, \qquad i'_t = \frac{i_t}{f_t + i_t}$$

Teraz $h_t$ to ważona średnia $h_{t-1}$ i $\tilde{h}_t$ — skala nie rośnie z $T$.

**Efekt uboczny:** bramka wyjściowa $o_t$ i osobny stan komórki $c_t$ stają się zbędne i są usuwane.

---

## 4. Finalna postać minLSTM

$$\boxed{h_t = f'_t \odot h_{t-1} + i'_t \odot \tilde{h}_t}$$

gdzie:

$$f_t = \sigma(\text{Linear}(x_t)), \quad i_t = \sigma(\text{Linear}(x_t)), \quad \tilde{h}_t = \text{Linear}(x_t)$$
$$f'_t = \frac{f_t}{f_t + i_t}, \quad i'_t = \frac{i_t}{f_t + i_t}$$

Parametry: $O(3 d_h d_x)$ zamiast $O(4 d_h(d_x + d_h))$ — przy $d_h = 2d_x$ to **~25% parametrów LSTM**.

---

## 5. Parallel Scan — dlaczego można zrównoleglić

Rekurencja minLSTM ma postać:

$$h_t = a_t \odot h_{t-1} + b_t \quad \text{gdzie } a_t = f'_t,\ b_t = i'_t \odot \tilde{h}_t$$

To jest tzw. **liniowa rekurencja pierwszego rzędu** — znana rodzina funkcji rozwiązywalna przez algorytm *prefix scan* (Blelloch, 1990). Kluczowe: $a_t$ i $b_t$ zależą tylko od $x_t$, więc są znane z góry dla całej sekwencji.

Algorytm oblicza $h_{1:T}$ w $O(\log T)$ kroków na GPU zamiast $O(T)$ kroków sekwencyjnych.

**Przyspieszenie w praktyce** (T4 GPU, batch=64):
- $T = 512$: **235× szybsze** niż klasyczny LSTM
- $T = 4096$: **1361× szybsze**

### Log-space implementation

Naiwne mnożenie wielu $f'_t \in (0,1)$ prowadzi do numerical underflow w FP32. Rozwiązanie: obliczenia w przestrzeni logarytmicznej:

$$\log(f'_t \cdot f'_{t-1} \cdot \ldots) = \sum \log f'_t \quad \rightarrow \quad \texttt{torch.cumsum}(\log f'_t)$$

Człon addytywny $b_t$ sumuje się przez `torch.logcumsumexp` — numerycznie stabilną wersję $\log\sum\exp$.

---

## 6. Ważna uwaga: minimalna liczba warstw

Przy **1 warstwie** bramki $f'_t$ i $i'_t$ zależą od $x_t$ — wejścia niezależnego od czasu (tokeny są stałe). Bramki są więc time-independent: model ma ograniczoną zdolność modelowania złożonych zależności.

Przy **2+ warstwach** wyjście pierwszej warstwy $h^{(1)}_{1:T}$ staje się time-dependent wejściem dla drugiej warstwy. Bramki w warstwie 2 są już zależne od czasu — model odzyskuje pełną ekspresywność.

**Wniosek:** minLSTM wymaga co najmniej 2 warstw dla nietrywialnych zadań (potwierdzone eksperymentalnie w papierze na Selective Copying Task: 1 warstwa → 37% accuracy, 2 warstwy → 86%, 3 warstwy → 96%).

---

## 7. Podsumowanie

minLSTM to radykalne uproszczenie LSTM osiągnięte przez trzy zmiany: usunięcie $h_{t-1}$ z bramek, usunięcie `tanh`, i normalizację bramek. Efektem jest komórka rekurencyjna z 25% parametrów LSTM, uczona równolegle przez parallel scan (235× szybciej dla $T=512$), która według papera osiąga porównywalną jakość do LSTM, GRU i Mamby na standardowych benchmarkach.

---

## 8. Q&A

**Q: Czy to jest modyfikacja architektury?**  
Tak — modyfikacja struktury komórki rekurencyjnej, która jednocześnie zmienia sposób uczenia (eliminuje BPTT na rzecz parallel scan).

---

**Q: Czy w minLSTM nadal są bramki forget i input?**  
Tak. Bramki $f_t$ i $i_t$ zostają — zmienia się tylko od czego zależą. W LSTM zależały od $[x_t, h_{t-1}]$, w minLSTM zależą tylko od $x_t$. Odpada natomiast bramka wyjściowa $o_t$ oraz osobny stan komórki $c_t$.

---

**Q: Skoro $h_{t-1}$ zostało usunięte z bramek, to dlaczego we wzorze $h_t = f'_t \odot h_{t-1} + i'_t \odot \tilde{h}_t$ nadal jest $h_{t-1}$?**  
To są dwie różne role $h_{t-1}$:
- **W bramkach** — służył jako wejście do obliczania proporcji. To zostało usunięte.
- **W rekurencji** — to sama pamięć, którą mieszamy z nowym kandydatem. Tego nie można usunąć, bo to definicja sieci rekurencyjnej.

Parallel scan nie usuwa $h_{t-1}$ z rekurencji — rozwiązuje całą rekurencję algebraicznie naraz, bez sekwencyjnego czekania.

---

**Q: Skoro $h_{t-1}$ jest w rekurencji, to jak parallel scan działa bez czekania?**  
Rozwijając rekurencję $h_t = a_t h_{t-1} + b_t$ analitycznie, każde $h_t$ można wyrazić wyłącznie przez $a_1,\ldots,a_t$ i $b_1,\ldots,b_t$ bez żadnego $h_{t-1}$. To są iloczynu i sumy skumulowane, które GPU liczy równolegle przez `cumsum`. Warunek konieczny: $a_t$ i $b_t$ muszą być znane z góry — stąd krok 1 uproszczenia.

---

**Q: Co się dzieje z bramką wyjściową $o_t$?**  
Odpada jako efekt uboczny kroku 3. W LSTM $o_t$ kontrolowało skalę wyjścia przez $h_t = o_t \odot \tanh(c_t)$. Po normalizacji $f'_t + i'_t = 1$ rekurencja jest ważoną średnią — skala jest kontrolowana automatycznie i $o_t$ jest zbędne. Przy okazji $h_t = c_t$, więc osobny stan komórki $c_t$ też odpada.

---

**Q: Czy $x_t$ to cały tekst czy pojedyncze słowo?**  
$x_t$ to **pojedynczy token** w kroku czasowym $t$, przekształcony przez `nn.Embedding` w wektor np. 128-wymiarowy. Cała recenzja IMDB to sekwencja $x_1, x_2, \ldots, x_T$. Podczas uczenia (parallel mode) wszystkie tokeny wchodzą naraz jako tensor `[batch_size, seq_len, embed_dim]`.

---

**Q: Dlaczego potrzeba co najmniej 2 warstw, skoro pamięć długotrwała jest już w 1 warstwie przez rekurencję?**  
Pamięć w czasie ($h_{t-1}$ w rekurencji) jest w warstwie 1 — to prawda. Problem leży gdzie indziej: bramki w warstwie 1 widzą tylko $x_t$ (embedding słowa), który jest **zawsze taki sam** niezależnie od kontekstu. Bramka dla słowa "great" podejmie tę samą decyzję czy poprzedza je "was" czy "Not". Od warstwy 2 wejściem jest $h^{(1)}_t$ zawierający skompresowaną historię — bramki stają się context-aware. Wynik: 1 warstwa → 37% accuracy, 2 warstwy → 86% na Selective Copying Task.