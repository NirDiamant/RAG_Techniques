# Contributing to RAG Techniques

Welcome to the world's largest and most comprehensive repository of Retrieval-Augmented Generation (RAG) tutorials! 🌟 We're thrilled you're interested in contributing to this ever-growing knowledge base. Your expertise and creativity can help us maintain our position at the forefront of RAG technology.

## Join Our Community

We have a vibrant Discord community where contributors can discuss ideas, ask questions, and collaborate on RAG techniques. Join us at:

[RAG Techniques Discord Server](https://discord.gg/cA6Aa4uyDX)

Don't hesitate to introduce yourself and share your thoughts!

## Ways to Contribute

We welcome contributions of all kinds! Here are some ways you can help:

1. **Add New RAG Techniques:** Create new notebooks showcasing novel RAG methods.
2. **Improve Existing Notebooks:** Enhance, update, or expand our current tutorials.
3. **Fix Bugs:** Help us squash bugs in existing code or explanations.
4. **Enhance Documentation:** Improve clarity, add examples, or fix typos in our docs.
5. **Share Creative Ideas:** Have an innovative idea? We're all ears!
6. **Engage in Discussions:** Participate in our Discord community to help shape the future of RAG.

Remember, no contribution is too small. Every improvement helps make this repository an even better resource for the community.

## Reporting Issues

Found a problem or have a suggestion? Please create an issue on GitHub, providing as much detail as possible. You can also discuss issues in our Discord community.

## Contributing Code or Content

1. **Fork and Branch:** Fork the repository and create your branch from `main`.
2. **Make Your Changes:** Implement your contribution, following our best practices.
3. **Test:** Ensure your changes work as expected.
4. **Follow the Style:** Adhere to the coding and documentation conventions used throughout the project.
5. **Commit:** Make your git commits informative and concise.
6. **Stay Updated:** The main branch is frequently updated. Before opening a pull request, make sure your code is up-to-date with the current main branch and has no conflicts.
7. **Push and Pull Request:** Push to your fork and submit a pull request.
8. **Discuss:** Use the Discord community to discuss your contribution if you need feedback or have questions.

## Adding a New RAG Method

When adding a new RAG method to the repository, please follow these additional steps:

1. Create your notebook in the `all_rag_techniques` folder.
2. Update BOTH the list and table in README.md:

   ### A. Update the List of Techniques
   - Add your new method to the list of techniques in the README
   - Place it in the appropriate position based on complexity (methods are sorted from easiest to most complicated)
   - Use the following format for the link:
     ```
     ### [Number]. [Your Method Name 🏷️](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/your_file_name.ipynb)
     ```
   - Replace `[Number]` with the appropriate number, `[Your Method Name]` with your method's name, and `your_file_name.ipynb` with the actual name of your notebook file
   - Choose an appropriate emoji that represents your method

   ### B. Update the Techniques Table
   - Add a new row to the table with your technique
   - Include all available implementations (LangChain, LlamaIndex, and/or Runnable Script)
   - Use the following format:
     ```
     | [Number] | [Category] | [LangChain](...) / [LlamaIndex](...) / [Runnable Script](...) | [Description] |
     ```
   - Make sure to:
     - Update the technique number to maintain sequential order
     - Choose the appropriate category with emoji
     - Include links to all available implementations
     - Write a clear, concise description

   ### C. Important Note
   - After inserting your new method, make sure to update the numbers of all subsequent techniques to maintain the correct order in BOTH the list and the table
   - The numbers in the list and table must match exactly
   - If you add a new technique as number 5, all techniques after it should be incremented by 1 in both places

For example, if you're adding a new technique between Simple RAG and Next Method:

In the list:
```
### 1. [Simple RAG 🌱](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb)
### 2. [Your New Method 🆕](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/your_new_method.ipynb)
### 3. [Next Method 🔜](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/next_method.ipynb)
```

And in the table:
```
| 1 | Foundational 🌱 | [LangChain](...) / [LlamaIndex](...) / [Runnable Script](...) | Basic RAG implementation |
| 2 | Your Category 🆕 | [LangChain](...) / [LlamaIndex](...) / [Runnable Script](...) | Your new method description |
| 3 | Next Category 🔜 | [LangChain](...) / [LlamaIndex](...) / [Runnable Script](...) | Next method description |
```

Remember: Always update BOTH the list and table when adding new techniques, and ensure the numbers match exactly between them.

## Notebook Structure

For new notebooks or significant additions to existing ones, please follow this structure:

1. **Title and Overview:** Clear title and brief overview of the technique.

2. **Detailed Explanation:** Cover motivation, key components, method details, and benefits.

3. **Visual Representation:** Include a diagram to visualize the technique. We recommend using Mermaid syntax for creating these diagrams. Here's how to do it:

• Create a graph using Mermaid's graph TD (top-down) syntax<br>
• You can use Claude or other AI assistants to help you design the graph if needed<br>
• Paste your Mermaid code into [Mermaid Live Editor](https://mermaid.live/)<br>
• In the "Actions" tab of Mermaid Live Editor, download the SVG file of your diagram<br>
• Store the SVG file in the [images folder](https://github.com/NirDiamant/RAG_Techniques/tree/main/images) of the repository<br>
• Use an appropriate, descriptive name for the file<br>
• In your notebook, display the image using Markdown syntax:<br>
  ```markdown
  ![Your Technique Name](../images/your-technique-name.svg)
  ```

This process ensures consistency in our visual representations and makes it easy for others to understand and potentially modify the diagrams in the future.

4. **Implementation:** Step-by-step Python implementation with clear comments and explanations.

5. **Usage Example:** Demonstrate the technique with a practical example.

6. **Comparison:** Compare with basic RAG, both qualitatively and quantitatively if possible.

7. **Additional Considerations:** Discuss limitations, potential improvements, or specific use cases.

8. **References:** Include relevant citations or resources if you have.

## Notebook Best Practices

To ensure consistency and readability across all notebooks:

1. **Code Cell Descriptions:** Each code cell should be preceded by a markdown cell with a clear, concise title describing the cell's content or purpose.

2. **Clear Unnecessary Outputs:** Before committing your notebook, clear all unnecessary cell outputs. This helps reduce file size and avoids confusion from outdated results.

3. **Consistent Formatting:** Maintain consistent formatting throughout the notebook, including regular use of markdown headers, code comments, and proper indentation.

## Code Quality and Readability

To ensure the highest quality and readability of our code:

1. **Write Clean Code:** Follow best practices for clean, readable code.
2. **Use Comments:** Add clear and concise comments to explain complex logic.
3. **Format Your Code:** Use consistent formatting throughout your contribution.
4. **Language Model Review:** After completing your code, consider passing it through a language model for additional formatting and readability improvements. This extra step can help make your code even more accessible and maintainable.

## Documentation

Clear documentation is crucial. Whether you're improving existing docs or adding new ones, follow the same process: fork, change, test, and submit a pull request.

## Final Notes

We're grateful for all our contributors and excited to see how you'll help expand the world's most comprehensive RAG resource. Don't hesitate to ask questions in our Discord community if you're unsure about anything.

Let's harness our collective knowledge and creativity to push the boundaries of RAG technology together!

Happy contributing! 🚀

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=contributing-guide)