# RH Mudança Emprego - AIDAPT

nas métricas de sucesso (6), acrescentei isto: 

O problema é de classificação binária (sim/não) e a nossa métrica principal seria a de recall pois mede quantos candidatos que realmente procuram mudar foram encontrados (um falso positivo não é tão grave como um falso negativo - falhar um candidato recetivo (falso negativo) é uma oportunidade perdida, e contactar alguém que não quer mudar (falso positivo) custa apenas um contacto).

O F1-Score junta precision e recall numa só nota e evita que o modelo “ganhe” só por prever “Sim” para toda a gente.
A Matriz de Confusão mostra quantos erros de cada tipo existem, ligando-os ao custo para o negócio. 
Accuracy não é muito fiável como métrica principal (apenas 25% dos candidatos procuram mudar, por isso, um modelo que prevesse sempre "não" teria cerca de 75% de accuracy sem aprender nada).
