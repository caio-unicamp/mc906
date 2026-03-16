# mc906
Repositório para realização dos trabalhos para a disciplina MC906 de introdução a inteligência artificial na Unicamp ministrada pelo professor Anderson de Rezende Rocha

## Reversi / Othello em Pygame

Interface completa do jogo Reversi (Othello), fiel às regras clássicas:

- Tabuleiro 8x8 com formação inicial correta;
- Jogadas legais com captura em 8 direções;
- Destaque visual de jogadas válidas;
- Passe automático de turno quando não há jogadas possíveis;
- Fim de jogo com contagem final e vencedor;
- Reinício rápido do jogo.

### Como executar

1. Instale as dependências:

	pip install -r requirements.txt

2. Execute o jogo:

	python reversi_pygame.py

### Controles

- Clique esquerdo: jogar peça em uma casa válida
- `R`: reiniciar partida
- `Q` ou `Esc`: sair
