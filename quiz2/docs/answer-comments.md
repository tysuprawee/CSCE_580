# Quiz 2 Responses

## Q1. Energy comparison

| Setting          | AI / LLM alternative            | Classical alternative | AI energy (Wh) | Classical energy (Wh) | Difference (AI − Classical) |
|------------------|---------------------------------|-----------------------|----------------|------------------------|-----------------------------|
| Generate Image   | Cloud AI (DALL·E, Midjourney)   | Local AI (On-device)  | 2.9            | 0.2                    | 2.7                         |
| Writing          | AI Writing Assistant            | Basic Text Editor     | 1.5            | 0.1                    | 1.4                         |
| Math             | AI Math Solver                  | Calculator App        | 1.2            | 0.1                    | 1.1                         |

**Average difference.** The mean energy gap across the three settings is \(\frac{2.7 + 1.4 + 1.1}{3} = 1.73\) Wh, so the AI/LLM options consume about **1.73 Wh more** than the classical approaches on average.

## Q2. Recipe conversion experiment

### AI test case (template)
- **Test Case ID:** R3-Recipes-20241005-01
- **Title:** Convert cookie and cake recipes to R3 JSON using prompt-full and prompt-partial strategies
- **Objective:** Ensure the conversational agent produces valid R3-compliant recipe JSON for two foods using multiple prompting strategies, scoring each output with the defined rubric.
- **Preconditions:**
  - Access to the original recipe texts saved in `./data/original_recipe1.txt` and `./data/original_recipe2.txt`.
  - Availability of the rubric-based evaluation script (`./code/goodness_score.py`).
- **Test Data:**
  - Recipe 1: Easy Oatmeal Chocolate Chip Cookies (Instructables).
  - Recipe 2: Ricotta Blueberry Cake with Streusel Crumble (Instructables).
- **Steps:**
  1. Prepare prompt-full (PF) prompts PF1–PF3 for each recipe. Capture the prompts and resulting JSON files in `./data`.
  2. Prepare prompt-partial (PP) prompts to extract ingredients and instructions separately for each recipe.
  3. Run `python ./code/pp_combine.py` to merge the partial outputs into final R3 JSON files.
  4. Execute `python ./code/goodness_score.py ./data` to score all JSON outputs.
  5. Record scores and observations in `./docs/answer-comments.md`.
- **Expected Result:**
  - JSON outputs load successfully.
  - Goodness scores reflect metadata completeness (higher for richer outputs, lower for sparse ones).
  - Partial combination script produces valid JSON that matches the R3 schema expectations.

### Data preparation notes
- Recipes were copied verbatim from Instructables into `original_recipe1.txt` and `original_recipe2.txt`. No additional cleaning was required beyond trimming duplicate headings.

### Prompting strategies and outputs
For each recipe, four conversions were produced. Prompt text is documented below, and the resulting files are stored in `./data`.

**Optional extension (Q3).** A fifth conversion per recipe was generated with a *chain-of-thought then structure* meta-prompt designed to elicit intermediate planning before formatting. The resulting JSON files are `oatmeal_chain_prompt.json` and `blueberry_chain_prompt.json`.

#### Easy Oatmeal Chocolate Chip Cookies
- **PF1 Prompt:** “Convert the following oatmeal chocolate chip cookie recipe into the R3 JSON format. Include recipe name, provenance, macronutrients, detailed ingredients with allergy data, boolean dietary flags, timing metadata, and instructions broken into single-action tasks with background knowledge.” → `oatmeal_pf1.json`
- **PF2 Prompt:** “Produce a concise R3 JSON for the same recipe, keeping ingredients and step tasks but omitting optional imagery. You may leave macronutrients empty if uncertain.” → `oatmeal_pf2.json`
- **PF3 Prompt:** “Return only the minimal JSON necessary to show the structure for this recipe without elaborating ingredients or steps.” → `oatmeal_pf3.json`
- **PP Prompts:**
  - **PP-1 (Ingredients):** “List the ingredients for the cookie recipe with quantities in a JSON object under an `ingredients` array following the R3 schema.” → `oatmeal_pp_ingredients.json`
  - **PP-2 (Instructions):** “Create R3-style instructions for the cookie recipe with single-action tasks and background knowledge, plus timing metadata.” → `oatmeal_pp_instructions.json`
  - Combined via `pp_combine.py` → `oatmeal_pp_combined.json`.

#### Ricotta Blueberry Cake with Streusel Crumble
- **PF1 Prompt:** “Transform the ricotta blueberry cake recipe into a full R3 JSON including provenance, macronutrients, detailed ingredients (with qualities and alternatives), and multi-step instructions.” → `blueberry_pf1.json`
- **PF2 Prompt:** “Generate a streamlined R3 JSON for the cake that keeps instructions and ingredients but allows leaving macronutrients blank.” → `blueberry_pf2.json`
- **PF3 Prompt:** “Provide a skeletal R3 JSON for the cake, focusing only on the primary cooking directive.” → `blueberry_pf3.json`
- **PP Prompts:**
  - **PP-1 (Ingredients):** “Output the cake ingredients as R3-compatible ingredient objects with quantity measures.” → `blueberry_pp_ingredients.json`
  - **PP-2 (Instructions):** “Write R3 instructions for the cake with action-specific tasks, plus prep/cook time and serving metadata.” → `blueberry_pp_instructions.json`
  - Combined via `pp_combine.py` → `blueberry_pp_combined.json`.

### Goodness score results
Scores were generated with the rubric-aligned script (`python ./code/goodness_score.py ./data`).

| Recipe & Prompt              | File                            | Score |
|-----------------------------|---------------------------------|-------|
| Oatmeal PF1                 | `oatmeal_pf1.json`              | 100   |
| Oatmeal PF2                 | `oatmeal_pf2.json`              | 90    |
| Oatmeal PF3                 | `oatmeal_pf3.json`              | 80    |
| Oatmeal PP (combined)       | `oatmeal_pp_combined.json`      | 100   |
| Oatmeal Chain Prompt (opt.) | `oatmeal_chain_prompt.json`     | 100   |
| Blueberry PF1               | `blueberry_pf1.json`            | 100   |
| Blueberry PF2               | `blueberry_pf2.json`            | 90    |
| Blueberry PF3               | `blueberry_pf3.json`            | 80    |
| Blueberry PP (combined)     | `blueberry_pp_combined.json`    | 100   |
| Blueberry Chain Prompt (opt.) | `blueberry_chain_prompt.json` | 100   |

### GAICO-style comparison (Q4 optional)
Using the lightweight `gaico_compare.py` helper (a command-line port of the GAICO similarity routine), I compared each trio of prompt-full outputs. The tool reports Jaccard similarity between the combined instruction and action strings.

```
python ./code/gaico_compare.py ./data/oatmeal_pf1.json ./data/oatmeal_pf2.json ./data/oatmeal_pf3.json
python ./code/gaico_compare.py ./data/blueberry_pf1.json ./data/blueberry_pf2.json ./data/blueberry_pf3.json
```

The oatmeal recipe shows PF1 vs PF2 similarity of 0.03 (shared structure) while PF3 diverges (0.00). The blueberry recipe scores 0.06 between PF1 and PF2, again highlighting PF3's sparse output.

### Answers
- **Q1. Which conversion approach performed better?** For both recipes, the prompt-partial pipeline (PP-1 + PP-2 combined via Python) matched the strongest prompt-full score (100) while offering clearer control over metadata completeness. Among the single prompts, PF1 also reached 100, but PF2/PF3 degraded as constraints tightened. Partial prompting therefore provides consistent high-quality results with transparent assembly.
- **Q2. Highest goodness score per food?**
  - Easy Oatmeal Chocolate Chip Cookies: 100 (PF1 and PP combined).
  - Ricotta Blueberry Cake with Streusel Crumble: 100 (PF1 and PP combined).
- **Q3. Optional extra prompting?** The chain-of-thought structured prompt achieved a perfect 100 for both recipes, matching the best full prompts while keeping the JSON compact.
- **Q4. Optional GAICO comparison?** The comparison tool revealed PF1 and PF2 share ~3–6% of instruction/action tokens, whereas PF3 overlaps 0% with the richer prompts, quantifying how the skeletal prompt omits critical actions.
