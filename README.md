# mc906
Repositório para realização dos trabalhos para a disciplina MC906 de introdução a inteligência artificial na Unicamp ministrada pelo professor Anderson de Rezende Rocha

## Reversi / Othello em Pygame

Implementação completa de Reversi/Othello com:

- motor de regras reutilizável;
- interface Pygame para jogo humano;
- agentes de IA com Minimax e Alpha-Beta;
- heurísticas plugáveis;
- torneio automatizado com exportação para CSV.

---

## Estrutura do projeto

- `src/othello_core.py`
	- núcleo do jogo (regras, movimentos válidos, aplicação de jogadas, estado terminal, placar);
	- classe `OthelloGame` reutilizada pela UI e pelos agentes.

- `src/interactive_pygame.py`
	- modo interativo humano vs humano em Pygame;
	- destaca jogadas válidas e mostra mensagens de passe/fim de jogo.

- `src/minimax_strategy.py`
	- estratégia Minimax;
	- versão fixa por profundidade e versão com limite de tempo (`minimax_timed_decision`).

- `src/alpha_beta_pruning_strategy.py`
	- estratégia Alpha-Beta pruning;
	- versão fixa por profundidade e versão com limite de tempo (`alphabeta_timed_decision`).

- `src/mobility_heuristic.py`
	- heurística de mobilidade.

- `src/border_control_heuristic.py`
	- heurística de controle de cantos.

- `src/frontier_heuristic.py`
	- heurística de frontier disks (peças adjacentes a casas vazias).

- `src/agents_tournament.py`
	- executa torneio entre combinações de estratégia+heurística;
	- opcionalmente mostra partidas em Pygame;
	- salva resultados em CSV.

---

## Estratégias de busca

### 1) Minimax

Arquivo: `src/minimax_strategy.py`

Ideia:
- assume adversário ótimo;
- alterna nós MAX (jogador atual) e MIN (oponente);
- usa heurística para avaliar folhas.

Recursos implementados:
- *iterative deepening* na versão com tempo;
- *move ordering* para explorar melhores lances primeiro;
- instrumentação por jogada:
	- nós expandidos;
	- profundidade atingida;
	- tempo de busca.

### 2) Alpha-Beta Pruning

Arquivo: `src/alpha_beta_pruning_strategy.py`

Ideia:
- equivalente ao Minimax em decisão final;
- elimina ramos que não podem melhorar a solução (podas via `alpha` e `beta`).

Recursos implementados:
- *iterative deepening* na versão com tempo;
- *move ordering* (fundamental para aumentar podas);
- instrumentação por jogada (nós, profundidade, tempo).

---

## Heurísticas

As heurísticas recebem `(board, player)` e retornam um escore (maior é melhor para `player`).

### 1) Mobility Heuristic

Arquivo: `src/mobility_heuristic.py`

Fórmula:

`mobility = my_moves - opp_moves`

Intuição:
- favorece posições com mais opções de jogada;
- reduz chance de ficar sem movimento.

### 2) Corner (Border Control) Heuristic

Arquivo: `src/border_control_heuristic.py`

Versão bruta:

`corner_score = my_corners - opp_corners`

Versão normalizada opcional:

`corner_score = (my_corners - opp_corners) / (my_corners + opp_corners + 1)`

Intuição:
- cantos são estáveis e muito fortes em Othello;
- controla bordas e reduz viradas futuras.

### 3) Frontier Heuristic

Arquivo: `src/frontier_heuristic.py`

Frontier disks = peças adjacentes a ao menos uma casa vazia.

Fórmula:

`frontier_score = opp_frontier - my_frontier`

Intuição:
- ter menos frontier disks é melhor;
- peças de fronteira são mais vulneráveis a viradas.

---

## Tournament de agentes

Arquivo: `src/agents_tournament.py`

O torneio testa automaticamente combinações de:
- estratégias: `minimax`, `alphabeta`;
- heurísticas: `mobility`, `corner`, `frontier`.

Para cada partida, o script registra:
- agentes (preto e branco);
- vencedor;
- placar final;
- diferença de discos;
- número de jogadas;
- duração da partida.

Também há logs no terminal com o jogo atual (`[START]` / `[END]`).

---

## Como executar

### 1) Instalar dependências

```bash
uv sync
```

### 2) Rodar modo interativo (Pygame)

```bash
uv run -m src.interactive_pygame
```

Controles:
- clique esquerdo: jogar em casa válida;
- `R`: reiniciar;
- `Q` ou `Esc`: sair.

### 3) Rodar torneio (headless)

```bash
uv run -m src.agents_tournament --repetitions 3 --time-limit 0.5 --max-depth 64 --output results/agent_tournament.csv
```

### 4) Rodar torneio mostrando partidas em Pygame

```bash
uv run -m src.agents_tournament --show-games --move-delay-ms 80 --repetitions 1 --time-limit 0.5 --output results/agent_tournament.csv
```

---

## Observações

- O limite de tempo por jogada é aplicado nas versões `*_timed_decision`.
- `Alpha-Beta` tende a explorar mais profundidade que `Minimax` no mesmo orçamento de tempo, principalmente com boa ordenação de movimentos.
- O CSV final facilita comparar desempenho médio entre combinações de estratégia e heurística.
	