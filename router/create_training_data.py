"""
Router Training Data Generator
================================

This script creates labeled training data for the router — the component
that decides which expert (Code, Math, or General) should handle a prompt.

How the Router Works (High-Level):
-----------------------------------
1. User sends a prompt: "Write a function to sort a list"
2. Router encodes the prompt into a vector (embedding) using sentence-transformers
3. Router classifies the embedding → "code"
4. System loads the Code Expert LoRA adapter
5. Model generates the response using the code expert

Why Embeddings + Classifier (not keyword matching)?
---------------------------------------------------
Simple keyword matching ("if 'code' in prompt → code expert") would fail on:
- "Explain how recursion works" (code topic, no code keywords)
- "What's the difference between a list and a tuple?" (code, but sounds general)
- "If I have 3 apples and buy 2 more..." (math, no math keywords)

Instead, we use semantic embeddings:
- all-MiniLM-L6-v2 converts text → 384-dimensional vector
- Similar meanings → similar vectors (even with different words)
- A classifier learns the decision boundary between expert domains

This approach is:
- More robust than keywords
- Lightweight (MiniLM is only 80MB, runs fast on CPU)
- Extensible (add new experts by adding labeled examples)

Research context:
- This is similar to how Switch Transformer (Fedus et al., 2022) routes tokens
- Our approach routes at the PROMPT level (coarser) rather than token level
- Prompt-level routing is simpler but still effective for distinct domains
"""

import json
import os


def create_router_training_data(output_dir: str = None):
    """
    Generate labeled examples for training the router classifier.

    Each example is: (prompt_text, expert_label)

    We create examples across 3 categories:
    - "code": Programming-related prompts
    - "math": Mathematical/numerical reasoning prompts
    - "general": Everything else

    Design principles:
    -------------------
    1. Include OBVIOUS examples (easy boundary learning)
    2. Include AMBIGUOUS examples (hard boundary learning)
    3. Balance across categories (avoid bias)
    4. Vary phrasing/style (generalization)

    We aim for ~100 examples per category (300 total).
    This is enough for a simple classifier because:
    - The embeddings already capture semantics well
    - We only have 3 classes
    - The domains are fairly distinct
    """

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    # =============================================
    # CODE EXPERT PROMPTS
    # =============================================
    # Mix of: direct coding requests, debugging, code explanation,
    # algorithm questions, language-specific queries
    code_prompts = [
        # Direct coding requests
        "Write a Python function to reverse a string",
        "Implement a binary search algorithm in Python",
        "Create a function that checks if a number is prime",
        "Write code to merge two sorted lists",
        "Implement a stack data structure using a list",
        "Write a Python class for a linked list",
        "Create a function to find duplicates in an array",
        "Write a recursive function to compute Fibonacci numbers",
        "Implement quicksort in Python",
        "Write a function to flatten a nested list",
        "Create a decorator that measures execution time",
        "Write a generator function for infinite prime numbers",
        "Implement a hash map from scratch",
        "Write code to validate an email address using regex",
        "Create a context manager for file handling",
        "Write a function to convert Roman numerals to integers",
        "Implement breadth-first search on a graph",
        "Write a Python script to read and parse a CSV file",
        "Create a function that returns the nth Fibonacci number",
        "Implement a basic calculator that handles +, -, *, /",

        # Debugging / fixing
        "Fix this code that gives an IndexError",
        "Debug this function that returns incorrect results",
        "Why does this code produce an infinite loop?",
        "What's wrong with this recursive function?",
        "Help me fix this TypeError in my Python script",

        # Code explanation
        "Explain how list comprehensions work in Python",
        "What does the yield keyword do?",
        "Explain the difference between append and extend in Python",
        "How do decorators work in Python?",
        "What is the difference between a class method and a static method?",
        "Explain how Python handles memory management",
        "What are metaclasses in Python?",
        "How does the GIL affect Python threading?",

        # Algorithm / data structure questions
        "What is the time complexity of binary search?",
        "Explain how a hash table handles collisions",
        "What's the difference between BFS and DFS?",
        "When should I use a heap vs a sorted list?",
        "Explain dynamic programming with an example",
        "What is memoization and when should I use it?",

        # Language-specific
        "How do I handle exceptions in Python?",
        "What are Python type hints and how do I use them?",
        "How do I use async/await in Python?",
        "What is the difference between a tuple and a list?",
        "How do virtual environments work in Python?",
        "Explain Python's __init__ and __new__ methods",

        # Practical coding tasks
        "Write a web scraper to extract article titles from a webpage",
        "Create a REST API endpoint using Flask",
        "Write a function to process JSON data from an API",
        "Implement rate limiting for API requests",
        "Write code to connect to a SQLite database",
        "Create a simple command-line todo app in Python",
        "Write a unit test for a sorting function",
        "Implement a simple chat server using sockets",
        "Write a function to download and save an image from URL",
        "Create a password generator with configurable options",

        # Ambiguous but code-related
        "How do I optimize my code for better performance?",
        "What design patterns should I use for this problem?",
        "Refactor this function to be more readable",
        "How do I make this code thread-safe?",
        "What's the best way to structure a Python project?",
        "How do I handle configuration files in Python?",
        "Write documentation for this function",
        "How do I profile my Python code?",
        "What's the most efficient way to read large files?",
        "How do I implement pagination in an API?",

        # More coding variations
        "Implement a trie data structure",
        "Write a function to detect cycles in a linked list",
        "Create a function to serialize a binary tree",
        "Implement LRU cache",
        "Write code to find the longest common subsequence",
        "Implement Dijkstra's algorithm",
        "Write a function to balance parentheses",
        "Create a simple regex engine",
        "Implement a thread pool",
        "Write a function to find all permutations of a string",
        "How do I implement a binary tree in Python?",
        "Write a Python function to count word frequencies in a text",
        "Implement merge sort with explanation",
        "Write code to detect if two strings are anagrams",
        "Create a function to generate all subsets of a set",
        "Write a Python program to solve the Tower of Hanoi",
        "Implement a priority queue using a heap",
        "Write code to find the shortest path in a maze",
        "Create a function to compress a string using run-length encoding",
        "Implement a bloom filter in Python",
    ]

    # =============================================
    # MATH EXPERT PROMPTS
    # =============================================
    # Mix of: word problems, calculations, algebra, geometry,
    # probability, logic puzzles
    math_prompts = [
        # Word problems (GSM8K style)
        "If a store sells apples at $3 each and you buy 7, how much do you spend?",
        "A train leaves at 9am traveling at 60mph. Another leaves at 10am at 80mph. When do they meet?",
        "Sarah has 24 cookies. She gives 1/3 to her friend. How many does she have left?",
        "A tank fills at 5 liters per minute. How long to fill a 300 liter tank?",
        "If you save $150 per month, how much will you have in 2 years?",
        "A rectangle's length is twice its width. If the perimeter is 36cm, find the dimensions.",
        "John earns $15/hour. He works 8 hours Monday-Friday. What's his weekly pay?",
        "A book costs $12.99 with a 20% discount. What's the final price?",
        "If 5 workers can build a wall in 10 days, how long for 8 workers?",
        "A car depreciates 15% per year. If it costs $20,000 now, what's it worth in 3 years?",

        # Basic arithmetic and algebra
        "Calculate 15% of 240",
        "Solve for x: 3x + 7 = 22",
        "What is 17 squared?",
        "Find the greatest common divisor of 48 and 36",
        "What is the least common multiple of 12 and 18?",
        "Simplify the fraction 84/126",
        "Calculate the compound interest on $1000 at 5% for 3 years",
        "Solve: 2x² - 8x + 6 = 0",
        "What is the square root of 169?",
        "Calculate 3/4 + 2/5",

        # Geometry
        "Find the area of a triangle with base 10 and height 6",
        "Calculate the circumference of a circle with radius 7",
        "What is the volume of a sphere with radius 3?",
        "Find the hypotenuse of a right triangle with legs 5 and 12",
        "Calculate the area of a trapezoid with parallel sides 8 and 12, height 5",
        "What is the surface area of a cube with side length 4?",
        "Find the angle in a right triangle if the opposite side is 3 and hypotenuse is 5",

        # Probability and statistics
        "What is the probability of rolling a 6 on a fair die?",
        "Calculate the mean and median of: 3, 7, 7, 2, 9, 4, 6",
        "If you flip a coin 3 times, what's the probability of getting all heads?",
        "What is the standard deviation of: 10, 12, 14, 16, 18?",
        "In how many ways can you arrange 5 books on a shelf?",
        "What's the probability of drawing a red card from a standard deck?",

        # Number theory and patterns
        "Is 97 a prime number?",
        "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        "Find the sum of all integers from 1 to 100",
        "How many factors does 72 have?",
        "What is 15 factorial?",
        "Find the first 10 terms of the Fibonacci sequence",

        # Applied math
        "Convert 72°F to Celsius",
        "How many combinations of 3 items from a set of 10?",
        "Calculate the distance between points (3,4) and (7,1)",
        "What is the slope of the line passing through (2,3) and (6,11)?",
        "Find the midpoint between (1,5) and (9,3)",
        "Calculate the percent increase from 80 to 100",

        # Equations and formulas
        "Solve the system: x + y = 10, x - y = 4",
        "What is the quadratic formula?",
        "Calculate the nth term of an arithmetic sequence with first term 3 and difference 5",
        "Find the sum of a geometric series with first term 2, ratio 3, and 5 terms",
        "Solve for x: log₂(x) = 5",

        # Logic and reasoning
        "If all A are B, and all B are C, are all A also C?",
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. What does the ball cost?",
        "How many handshakes occur in a room of 10 people if everyone shakes hands?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",

        # More math variations
        "What is the derivative of x³ + 2x²?",
        "Calculate the integral of 3x² from 0 to 4",
        "Find the limit of (x²-1)/(x-1) as x approaches 1",
        "What is 25% of 80?",
        "How many degrees in the interior angles of a pentagon?",
        "Calculate the area of a sector with radius 6 and angle 60°",
        "What is the binary representation of 42?",
        "Convert 0.375 to a fraction",
        "Find the determinant of matrix [[2,3],[4,5]]",
        "What is the dot product of vectors (1,2,3) and (4,5,6)?",
        "Solve: |2x - 3| = 7",
        "What is 8 to the power of 2/3?",
        "Calculate the perimeter of a regular hexagon with side 5",
        "How many diagonals does an octagon have?",
        "Find the inverse of the function f(x) = 2x + 3",
        "What is the probability of getting at least one 6 in 4 dice rolls?",
        "Calculate the weighted average of 70 (weight 2) and 90 (weight 3)",
        "Solve the inequality: 3x - 5 > 7",
        "What is the sum of the first 20 even numbers?",
        "Find all prime numbers between 50 and 70",
    ]

    # =============================================
    # GENERAL EXPERT PROMPTS
    # =============================================
    # Mix of: explanations, advice, creative, factual QA,
    # summarization, comparison, opinion
    general_prompts = [
        # Explanations
        "Explain how the internet works",
        "What is climate change and why does it matter?",
        "How does a car engine work?",
        "Explain the theory of relativity in simple terms",
        "What is blockchain technology?",
        "How do vaccines work?",
        "Explain the water cycle",
        "What causes earthquakes?",
        "How does GPS determine your location?",
        "Explain how airplanes stay in the air",

        # Factual Q&A
        "What are the seven continents?",
        "Who painted the Mona Lisa?",
        "What is the tallest mountain in the world?",
        "What causes the northern lights?",
        "How many planets are in our solar system?",
        "What is photosynthesis?",
        "Who invented the telephone?",
        "What is the largest ocean on Earth?",
        "What is the speed of light?",
        "How does the human heart work?",

        # Advice
        "How can I improve my public speaking skills?",
        "Give me tips for better time management",
        "What are some strategies for dealing with stress?",
        "How do I prepare for a job interview?",
        "What should I consider when buying a used car?",
        "Tips for starting a healthy morning routine",
        "How can I improve my memory?",
        "What are effective study techniques?",
        "How do I set and achieve long-term goals?",
        "Tips for maintaining a healthy work-life balance",

        # Creative
        "Write a haiku about autumn",
        "Describe a futuristic city",
        "Write a short motivational paragraph",
        "Create a metaphor for perseverance",
        "Write a brief story about discovering something unexpected",
        "Describe what freedom means to you",
        "Write a letter to your future self",
        "Create an analogy to explain teamwork",
        "Describe the perfect morning in a small town",
        "Write a short poem about the sea",

        # Comparison / Analysis
        "What are the pros and cons of remote work?",
        "Compare renewable and non-renewable energy sources",
        "What's the difference between a democracy and a republic?",
        "Compare the benefits of reading books vs watching documentaries",
        "What are the advantages and disadvantages of social media?",
        "Compare traditional education with online learning",
        "What's the difference between weather and climate?",
        "Compare electric cars vs gasoline cars",

        # History and culture
        "What were the main causes of World War I?",
        "Explain the significance of the Renaissance",
        "What was the Industrial Revolution?",
        "Describe the major achievements of ancient Egypt",
        "What was the Cold War about?",
        "Explain the impact of the printing press on society",

        # Science (non-math)
        "What is DNA and why is it important?",
        "How do black holes form?",
        "What is the difference between a virus and a bacteria?",
        "How does evolution work?",
        "What are the layers of the Earth?",
        "Explain the greenhouse effect",

        # Practical / How-to
        "How do I write an effective email?",
        "What should a good resume include?",
        "How do I start learning a new language?",
        "What are the steps to start a small business?",
        "How do I take better photographs?",
        "What's the best way to learn to cook?",

        # Opinion / Discussion
        "What makes a good leader?",
        "Why is reading important?",
        "What are the qualities of a good teacher?",
        "Why is critical thinking important in daily life?",
        "What role does art play in society?",
        "Why is diversity important in the workplace?",

        # More general variations
        "Summarize the key events of the French Revolution",
        "What are the benefits of regular exercise?",
        "Explain how a democracy works",
        "What is emotional intelligence?",
        "Describe the process of making cheese",
        "What are the different types of clouds?",
        "How does a refrigerator work?",
        "What is the scientific method?",
        "Explain the concept of supply and demand",
        "What are the main branches of philosophy?",
        "How does the stock market work?",
        "What is artificial intelligence?",
        "Explain the difference between empathy and sympathy",
        "What are the stages of grief?",
        "How does the immune system work?",
        "What is the butterfly effect?",
        "Explain what inflation is and how it affects people",
        "What are the benefits of meditation?",
        "How does WiFi work?",
        "What makes a good book recommendation?",
    ]

    # =============================================
    # BUILD LABELED DATASET
    # =============================================
    training_data = []

    for prompt in code_prompts:
        training_data.append({"text": prompt, "label": "code"})

    for prompt in math_prompts:
        training_data.append({"text": prompt, "label": "math"})

    for prompt in general_prompts:
        training_data.append({"text": prompt, "label": "general"})

    # Print statistics
    print(f"Router training data created:")
    print(f"  Code prompts: {len(code_prompts)}")
    print(f"  Math prompts: {len(math_prompts)}")
    print(f"  General prompts: {len(general_prompts)}")
    print(f"  Total: {len(training_data)}")

    # Save to JSON
    output_path = os.path.join(output_dir, "router_training_data.json")
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)
    print(f"\nSaved to: {output_path}")

    # Preview
    print("\nSample entries:")
    for label in ["code", "math", "general"]:
        samples = [d for d in training_data if d["label"] == label]
        print(f"\n  [{label}]:")
        for s in samples[:3]:
            print(f"    - {s['text']}")

    return training_data


if __name__ == "__main__":
    create_router_training_data()
