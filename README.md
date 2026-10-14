# RH Mudança Emprego - AIDAPT

nas métricas de sucesso (6), acrescentei isto: 

O problema é de classificação binária (sim/não) e a nossa métrica principal seria a de recall pois mede quantos candidatos que realmente procuram mudar foram encontrados (um falso positivo não é tão grave como um falso negativo - falhar um candidato recetivo (falso negativo) é uma oportunidade perdida, e contactar alguém que não quer mudar (falso positivo) custa apenas um contacto).

O F1-Score junta precision e recall numa só nota e evita que o modelo “ganhe” só por prever “Sim” para toda a gente.
A Matriz de Confusão mostra quantos erros de cada tipo existem, ligando-os ao custo para o negócio. 
Accuracy não é muito fiável como métrica principal (apenas 25% dos candidatos procuram mudar, por isso, um modelo que prevesse sempre "não" teria cerca de 75% de accuracy sem aprender nada).


8 - Recursos, riscos e restrições:
**Resposta:**

Um ficheiro CSV, com 19 158 candidatos e 14 varáveis (2 numéricas e restantes categóticas).
Cada linha é um candidato distinto, sem linhas duplicadas.

Qualidade dos dados:

- valores em falta em 8 colunas, com destaque para `tipo_empresa` (32%), `dimensao_empresa` (31%) e `genero`(24%) - estes valores em falta podem significar que as pessoas estão desempregadas, por exemplo.
- variaveis ordinais guardadas como texto - `anos_experiencia`, `dimensao_empresa`, `anos_desde_ultimo_emprego`, que terão de ser convertidas
-`nivel_educacao` é texto mas tem ordem natural (Ensino Básico < Secundário < Licenciatura < Mestrado < Doutoramento), o que pode ser aproveitado na codificação...
- variável_alvo desbalanceada (cerca de 25% de 'sim')

Riscos:

- o genero poderá ser uma variável sensível o que pode levar o modelo a tratar grupos de forma desigual
- em termos de enviesamento, este dataset é referente a pessoas que realizaram uma formação, não representando assim a população geral (não podemos generalizar conclusões)

9 Impacto Esperado

**Resposta:**


- Apoio à tomada de decisão: a equipa de RH/recrutamento passam a ter uma lista priorizada de candidatos, ordenada pela probabilidade de mudar de emprego
- Redução de risco: reduzir o risco de perder candidatos recetivos (falsos negativos)e o de investir esforço em quem não está interessado.
- Otimização de recursos: poupa tempo e custos de recrutamento (menos contactos sem resposta)
- Identificação de padrões relevantes: mostra que perfis estão mais abertos a mudar 
- Criação de vantagem competitiva: quem abordar primeiro os candidatos recetivos, tem mais hipotese de os contratar, antes da concorrencia

10 - Sintese 

Problema: uma entidade de recrutamento pretende identificar, entre os candidatos que frequentaram uma formação, aqueles que procuram mudar de emprego

Decisão a apoiar: a que candidatos fazer uma proposta de trabalho

Tipo de problema: classificação binária, com aprendizagem supervisionada, sobre 19.158 candidatos descritos por 14 variáveis (incluindo o alvo e o identificador)

Variável-alvo: `procura_mudanca_emprego` (Sim / Não), com cerca de 25% de “Sim”.

Métrica principal: recall da classe “Sim”, complementado pelo F1-score e pela matriz de confusão. A accuracy não é usada como métrica principal, por causa do desbalanceamento das classes.

**Critério de sucesso: xxxxxxxxxxxxxxxxxxx**
