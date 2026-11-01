# Laboratory Experiment 04: Knowledge Representation & Logical Reasoning

**Course:** IC-709D / IC-703 C | Artificial Intelligence Laboratory  
**Unit 4:** Knowledge Representation & Logical Reasoning (CO1, CO3)  
**Student Name:** Divyansh Verma  
**Roll No:** IC-2K23-31  

---

## 1. Aim / Problem Statement

To implement and evaluate fundamental knowledge representation paradigms in Artificial Intelligence:
1. **Semantic Networks:** Graph-based conceptual representations with property inheritance reasoning.
2. **Frames:** Object-centered slot-and-facet structural representations with default values and class inheritance.
3. **Rule-Based Systems:** Production systems using forward chaining (data-driven) and backward chaining (goal-driven) inference engines for domain advisory decision-making.
4. **Comparative Analysis:** Critically evaluate the expressiveness, computational complexity, and inference mechanisms across the three knowledge representation schemes.

---

## 2. Theoretical Background & Algorithms

Knowledge Representation (KR) is the formal study of how an autonomous agent represents real-world entities, concepts, rules, and facts so that inference engines can derive new knowledge and make intelligent decisions.

### 2.1 Paradigm 1: Semantic Networks (Quillian, 1968)
A **Semantic Network** represents declarative knowledge as a directed graph $G = (V, E)$:
- **Vertices ($V$):** Real-world concepts, entities, or instances (e.g., `UniversityPerson`, `Faculty`, `Student`, `Professor`).
- **Directed Edges ($E$):** Semantic relations connecting concepts, primarily:
  - `IS_A`: Taxonomic generalization/subsumption ($A \text{ IS\_A } B \implies A \subseteq B$).
  - Associative links: Domain relations such as `TEACHES`, `ENROLLED_IN`, `ASSISTS`.

**Inheritance Algorithm:**
To evaluate query $\text{Property}(x, p)$:
1. If $p$ is directly attached to node $x$, return $\text{value}(x, p)$.
2. Else, traverse all outgoing `IS_A` links $y \in \text{Parents}(x)$ recursively.
3. If no ancestor contains $p$, return `None` (open-world / absent property).

---

### 2.2 Paradigm 2: Frame-Based Systems (Minsky, 1975)
A **Frame** is a record-like, object-centered data structure representing a stereotypical concept or situation:
$$\text{Frame} = \langle \text{Name}, \text{Parent}, \{\text{Slot}_i \to \langle \text{Value}_i, \text{Facets}_i \rangle\} \rangle$$
- **Slots:** Attributes representing features or relationships (e.g., `instructor`, `credits`, `attendance_threshold`).
- **Facets:** Sub-attributes specifying data constraints, default values, or procedural attachments (`if-needed`, `if-added`).
- **Frame Inheritance:** Sub-frames inherit attributes from super-frames; local slot assignments override inherited values (polymorphic specialization).

---

### 2.3 Paradigm 3: Production Rule Systems & Inference Engines
A **Production System** represents procedural and heuristic domain expertise through IF-THEN rules:
$$\text{Rule } R_k: \quad \text{IF } \bigwedge_{j} c_{k,j} \quad \text{THEN } a_k$$

A rule-based architecture consists of:
1. **Knowledge Base (Rule Base):** Static set of domain rules.
2. **Working Memory (Fact Base):** Dynamic set of known facts for the current problem instance.
3. **Inference Engine:** Executes reasoning cycles via two modes:

#### Forward Chaining (Data-Driven Inference)
- **Direction:** Facts $\longrightarrow$ Goals.
- **Algorithm:**
  1. Initialize $\text{Inferred} \leftarrow \text{WorkingMemory}$.
  2. In each cycle, find all un-fired rules whose antecedents $\subseteq \text{Inferred}$.
  3. Add their consequents to $\text{Inferred}$.
  4. Repeat until a fixed point is reached (no new facts can be deduced).

#### Backward Chaining (Goal-Driven Inference)
- **Direction:** Goal / Hypothesis $\longrightarrow$ Facts.
- **Algorithm:**
  1. To verify hypothesis $G$: If $G \in \text{WorkingMemory}$, return `True`.
  2. Find all candidate rules concluding $G$.
  3. For each candidate rule, recursively evaluate all antecedent sub-goals.
  4. If all antecedents of any candidate rule hold, conclude $G$ is `True`; otherwise, backtrack.

---

## 3. Implementation Details

The laboratory implementation is structured within [`Lab4/`](.):
```
Lab4/
├── knowledge_representation.py      # Core modular Python implementation
├── outputs/
│   └── semantic_network.png         # Generated semantic network graph
└── README.md                        # Complete laboratory documentation
```

### Module Breakdown:
1. `SemanticNetwork`: Adjacency-list graph model with recursive taxonomic lookup and custom Matplotlib visualizer.
2. `Frame`: Hierarchical object structure supporting local values, default facets, and parent delegation.
3. `RuleBasedExpertSystem`: Academic Advisory Engine equipped with both `forward_chain()` and `backward_chain()` methods.

---

## 4. Results & Terminal Output

Execution trace obtained from running `python Lab4/knowledge_representation.py`:

```text
======================================================================
  LAB 4: KNOWLEDGE REPRESENTATION & REASONING (UNIT 4)
======================================================================

--- 1. SEMANTIC NETWORK KNOWLEDGE BASE ---
Inheritance Queries:
  * Query 'campus_access' for [UndergradStudent]: Granted (Inherited from: [UniversityPerson])
  * Query 'degree' for [UndergradStudent]: B.Tech (Inherited from: [UndergradStudent])
  * Query 'campus_access' for [Professor]: Granted (Inherited from: [UniversityPerson])
  * Query 'voting_rights' for [Professor]: Senate_Member (Inherited from: [Professor])
[Semantic Network] Diagram generated and saved to: Lab4/outputs/semantic_network.png

--- 2. FRAME-BASED KNOWLEDGE REPRESENTATION ---

Displaying Frame Structure for 'IC_709D_AI_Lab':
Frame: [IC_709D_AI_Lab] (Parent: LabCourse)
  * Slot: course_title    = Artificial Intelligence Laboratory
  * Slot: instructor      = Dr. Faculty
  * Slot: credits         = 2

Slot Inheritance Resolution:
  * Slot [course_title        ]: Artificial Intelligence Laboratory (Resolved from: IC_709D_AI_Lab)
  * Slot [practical_exam      ]: Mandatory                 (Resolved from: LabCourse)
  * Slot [attendance_threshold]: 75%                       (Resolved from: Course)
  * Slot [institution         ]: Engineering Institute     (Resolved from: Course)

--- 3. RULE-BASED EXPERT SYSTEM (ACADEMIC ADVISORY) ---
Initial Working Memory (Student Facts): ['cgpa_above_8_5', 'attendance_above_85', 'no_disciplinary_action', 'published_paper']

--- [Forward Chaining Execution Trace] ---
  Cycle 1: Fired 'R1_High_CGPA' -> Inferred fact: 'academically_distinguished'
  Cycle 1: Fired 'R2_Research' -> Inferred fact: 'research_competent'
  Cycle 1: Fired 'R3_Conduct' -> Inferred fact: 'good_standing'
  Cycle 1: Fired 'R4_Merit_Scholarship' -> Inferred fact: 'eligible_for_merit_scholarship'
  Cycle 1: Fired 'R5_Dean_Honor_List' -> Inferred fact: 'awarded_dean_honor_roll'
Total facts derived: 9 (Fired 5 rules)

Final Inferred Facts:
  + academically_distinguished
  + attendance_above_85
  + awarded_dean_honor_roll
  + cgpa_above_8_5
  + eligible_for_merit_scholarship
  + good_standing
  + no_disciplinary_action
  + published_paper
  + research_competent

--- Backward Chaining Query: Can we prove 'awarded_dean_honor_roll'? ---
? Testing Goal: 'awarded_dean_honor_roll'
Trying Rule [R5_Dean_Honor_List]: Requires {'eligible_for_merit_scholarship', 'research_competent'}
  ? Testing Goal: 'eligible_for_merit_scholarship'
  Trying Rule [R4_Merit_Scholarship]: Requires {'academically_distinguished', 'good_standing'}
    ? Testing Goal: 'academically_distinguished'
    Trying Rule [R1_High_CGPA]: Requires {'cgpa_above_8_5'}
      ? Testing Goal: 'cgpa_above_8_5'
      -> Confirmed directly from known facts.
    -> Rule [R1_High_CGPA] SUCCEEDED! Verified 'academically_distinguished'
    ? Testing Goal: 'good_standing'
    Trying Rule [R3_Conduct]: Requires {'attendance_above_85', 'no_disciplinary_action'}
      ? Testing Goal: 'attendance_above_85'
      -> Confirmed directly from known facts.
      ? Testing Goal: 'no_disciplinary_action'
      -> Confirmed directly from known facts.
    -> Rule [R3_Conduct] SUCCEEDED! Verified 'good_standing'
  -> Rule [R4_Merit_Scholarship] SUCCEEDED! Verified 'eligible_for_merit_scholarship'
  ? Testing Goal: 'research_competent'
  Trying Rule [R2_Research]: Requires {'published_paper'}
    ? Testing Goal: 'published_paper'
    -> Confirmed directly from known facts.
  -> Rule [R2_Research] SUCCEEDED! Verified 'research_competent'
-> Rule [R5_Dean_Honor_List] SUCCEEDED! Verified 'awarded_dean_honor_roll'

Hypothesis 'awarded_dean_honor_roll' Proven: True
```

---

## 5. Performance Analysis & Comparative Study

The table below contrasts the three primary knowledge representation paradigms across core architectural criteria:

| Evaluation Dimension | Semantic Networks | Frames | Rule-Based Systems |
| :--- | :--- | :--- | :--- |
| **Primary Data Structure** | Directed Graph (Nodes & Labeled Arcs) | Nested Records (Slots, Facets, Defaults) | Production Rules (IF-THEN Antecedent-Consequent) |
| **Knowledge Classification** | Structural / Declarative / Associative | Object-Centered / Situational | Procedural / Heuristic |
| **Reasoning Mechanism** | Path Traversal & Property Inheritance | Default Delegation & Slot Lookups | Forward & Backward Chaining |
| **Default Reasoning** | Limited / Hard to represent exceptions | Native support via Default Facets | Modeled via negation-as-failure rules |
| **Modularity & Scalability** | Moderate (Tangled graphs risk circularity) | High (Hierarchical encapsulation) | Very High (Rules can be added independently) |
| **Computational Overhead** | $O(V + E)$ graph traversal per query | $O(D)$ where $D$ is inheritance depth | $O(R \cdot F)$ per matching cycle |
| **Best Application Domain** | Taxonomies, Ontologies, Associative Memory | Knowledge Bases, Entity Modeling | Medical Diagnostics, Advisory Systems |

---

## 6. Graphical Visualizations

The semantic network graph constructed for the University Academic domain is displayed below:

![Semantic Network Knowledge Base](outputs/semantic_network.png)

### Architectural Observations:
- **Red Arrows (`IS_A`)**: Represent strict ontological inheritance paths (`Professor` $\to$ `Faculty` $\to$ `UniversityPerson`).
- **Blue Arrows (`Domain Relations`)**: Capture behavioral interactions across entities (e.g., `Professor` $\xrightarrow{\text{TEACHES}}$ `IC_709D_Lab`, `TeachingAssistant` $\xrightarrow{\text{ASSISTS}}$ `Faculty`).
- **Inheritance Traversal**: Queries targeting `UndergradStudent` successfully resolve higher-level properties like `campus_access: Granted` by traversing upward to `UniversityPerson`.

---

## 7. Learning Outcomes (CO1, CO3)

- **CO1 (Problem Formulation & Intelligent Problem Solving):** Formulated complex academic advisory and taxonomic domains into computable knowledge representations.
- **CO3 (Knowledge Representation & Reasoning):** Implemented property inheritance across graphs, slot-facet structures, and dual-direction inference engines (Forward & Backward Chaining).
- Analyzed tradeoffs between declarative associative graphs and procedural production systems.
- Mastered the design of modular, explainable AI reasoning frameworks without relying on black-box heuristics.
