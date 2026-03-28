# Contribuindo para RAG Techniques

Bem-vindo ao maior e mais abrangente repositório de tutoriais sobre Retrieval-Augmented Generation (RAG)! 🌟 Ficamos muito felizes com seu interesse em contribuir para esta base de conhecimento em constante crescimento. Sua experiência e criatividade ajudam a manter este projeto na fronteira da tecnologia RAG.

## Participe da Comunidade

Temos uma comunidade ativa no Discord onde os colaboradores podem discutir ideias, tirar dúvidas e colaborar em técnicas de RAG. Entre aqui:

[Servidor Discord do RAG Techniques](https://discord.gg/cA6Aa4uyDX)

Sinta-se à vontade para se apresentar e compartilhar suas ideias.

## Formas de Contribuir

Aceitamos contribuições de todos os tipos. Algumas formas de ajudar:

1. **Adicionar novas técnicas de RAG:** Crie novos notebooks mostrando métodos inéditos.
2. **Melhorar notebooks existentes:** Atualize, expanda ou refine os tutoriais atuais.
3. **Corrigir bugs:** Ajude a resolver problemas no código ou nas explicações.
4. **Aprimorar a documentação:** Melhore a clareza, adicione exemplos ou corrija erros de digitação.
5. **Compartilhar ideias criativas:** Tem uma ideia inovadora? Queremos ouvir.
6. **Participar das discussões:** Ajude a moldar o futuro do projeto na comunidade do Discord.

Nenhuma contribuição é pequena demais. Toda melhoria torna este repositório ainda mais útil para a comunidade.

## Relatando Problemas

Encontrou um problema ou tem uma sugestão? Abra uma issue no GitHub com o máximo de detalhes possível. Você também pode discutir o assunto no Discord.

## Contribuindo com Código ou Conteúdo

1. **Fork e branch:** Faça um fork do repositório e crie uma branch a partir de `main`.
2. **Faça suas alterações:** Implemente sua contribuição seguindo as melhores práticas do projeto.
3. **Teste:** Verifique se tudo funciona como esperado.
4. **Siga o estilo do projeto:** Respeite as convenções de código e documentação já usadas no repositório.
5. **Commit:** Escreva commits objetivos e informativos.
6. **Mantenha-se atualizado:** A branch principal é atualizada com frequência. Antes de abrir um pull request, garanta que seu código está alinhado com a `main` atual e sem conflitos.
7. **Push e pull request:** Envie sua branch e abra um pull request.
8. **Discuta quando necessário:** Use o Discord para pedir feedback ou esclarecer dúvidas.

## Adicionando um Novo Método de RAG

Ao adicionar um novo método de RAG ao repositório, siga também estes passos:

1. Crie seu notebook na pasta `all_rag_techniques`.
2. Atualize a lista e a tabela no `README.md`.

### A. Atualize a Lista de Técnicas

- Adicione o novo método à lista de técnicas no `README`.
- Coloque-o na posição correta com base na complexidade. Os métodos são organizados do mais simples ao mais avançado.
- Use o seguinte formato para o link:

```markdown
### [Number]. [Your Method Name 🏷️](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/your_file_name.ipynb)
```

- Substitua `[Number]` pelo número correto, `[Your Method Name]` pelo nome do método e `your_file_name.ipynb` pelo nome real do arquivo.
- Escolha um emoji apropriado para representar o método.

### B. Atualize a Tabela de Técnicas

- Adicione uma nova linha à tabela com sua técnica.
- Inclua todas as implementações disponíveis, como LangChain, LlamaIndex e/ou `Runnable Script`.
- Use o seguinte formato:

```markdown
| [Number] | [Category] | [LangChain](...) / [LlamaIndex](...) / [Runnable Script](...) | [Description] |
```

- Certifique-se de:
- Atualizar a numeração para manter a ordem sequencial.
- Escolher a categoria correta com emoji.
- Incluir links para todas as implementações disponíveis.
- Escrever uma descrição clara e concisa.

### C. Observação Importante

- Depois de inserir o novo método, atualize a numeração de todas as técnicas seguintes para manter a ordem correta tanto na lista quanto na tabela.
- Os números da lista e da tabela devem corresponder exatamente.
- Se você adicionar uma nova técnica na posição 5, todas as técnicas posteriores devem ser incrementadas em 1 nos dois lugares.

Por exemplo, se você estiver adicionando uma técnica entre `Simple RAG` e `Next Method`:

Na lista:

```markdown
### 1. [Simple RAG 🌱](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb)
### 2. [Your New Method 🆕](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/your_new_method.ipynb)
### 3. [Next Method 🔜](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/next_method.ipynb)
```

E na tabela:

```markdown
| 1 | Foundational 🌱 | [LangChain](...) / [LlamaIndex](...) / [Runnable Script](...) | Basic RAG implementation |
| 2 | Your Category 🆕 | [LangChain](...) / [LlamaIndex](...) / [Runnable Script](...) | Your new method description |
| 3 | Next Category 🔜 | [LangChain](...) / [LlamaIndex](...) / [Runnable Script](...) | Next method description |
```

Lembre-se: sempre atualize tanto a lista quanto a tabela ao adicionar novas técnicas e garanta que a numeração esteja idêntica nas duas.

## Estrutura dos Notebooks

Para novos notebooks ou grandes expansões em notebooks existentes, siga esta estrutura:

1. **Título e visão geral:** Título claro e breve apresentação da técnica.
2. **Explicação detalhada:** Cubra motivação, componentes principais, detalhes do método e benefícios.
3. **Representação visual:** Inclua um diagrama para visualizar a técnica. Recomendamos usar Mermaid para isso. O fluxo sugerido é:

• Crie um grafo usando a sintaxe `graph TD` do Mermaid<br>
• Você pode usar Claude ou outros assistentes de IA para ajudar a desenhar o grafo<br>
• Cole o código no [Mermaid Live Editor](https://mermaid.live/)<br>
• Na aba "Actions" do Mermaid Live Editor, baixe o arquivo SVG do diagrama<br>
• Armazene o SVG na [pasta `images`](https://github.com/NirDiamant/RAG_Techniques/tree/main/images) do repositório<br>
• Use um nome descritivo e apropriado para o arquivo<br>
• No notebook, exiba a imagem com Markdown:<br>

```markdown
![Your Technique Name](../images/your-technique-name.svg)
```

Esse processo ajuda a manter consistência visual e facilita a compreensão e manutenção futura dos diagramas.

4. **Implementação:** Passo a passo em Python com comentários e explicações claras.
5. **Exemplo de uso:** Demonstre a técnica com um caso prático.
6. **Comparação:** Compare com `Basic RAG`, de forma qualitativa e, se possível, quantitativa.
7. **Considerações adicionais:** Discuta limitações, melhorias possíveis ou casos de uso específicos.
8. **Referências:** Inclua citações ou recursos relevantes quando houver.

## Boas Práticas para Notebooks

Para manter consistência e legibilidade:

1. **Descrição das células de código:** Cada célula de código deve ser precedida por uma célula Markdown com um título curto e claro explicando o conteúdo ou objetivo da célula.
2. **Limpeza de saídas desnecessárias:** Antes de fazer commit do notebook, remova outputs desnecessários para reduzir o tamanho do arquivo e evitar confusão com resultados antigos.
3. **Formatação consistente:** Mantenha formatação uniforme em todo o notebook, incluindo uso regular de cabeçalhos Markdown, comentários no código e indentação adequada.

## Qualidade e Legibilidade do Código

Para garantir código de alta qualidade:

1. **Escreva código limpo:** Siga boas práticas para manter o código claro e legível.
2. **Use comentários:** Adicione comentários claros e objetivos para explicar lógicas mais complexas.
3. **Formate seu código:** Mantenha consistência de estilo em toda a contribuição.
4. **Revisão com modelo de linguagem:** Depois de terminar, considere passar o código por um modelo de linguagem para melhorar formatação e legibilidade. Isso pode tornar a contribuição ainda mais acessível e fácil de manter.

## Documentação

Documentação clara é essencial. Seja melhorando documentação existente ou criando conteúdo novo, siga o mesmo processo: faça o fork, altere, teste e envie um pull request.

## Observações Finais

Somos muito gratos por todos os colaboradores e estamos animados para ver como você vai ajudar a expandir o recurso mais completo sobre RAG. Se tiver qualquer dúvida, pergunte na comunidade do Discord.

Vamos usar nosso conhecimento e criatividade coletiva para avançar ainda mais a tecnologia RAG.

Boas contribuições! 🚀

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=contributing-guide)
