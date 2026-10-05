# React Frontend Developer Interview

## Entry Questions
- Walk me through how React's reconciliation algorithm (the diffing process) works under the hood.
- What is the difference between controlled and uncontrolled components? When would you use each?
- Explain the React component lifecycle — both class-based and the hooks equivalent.

## Core React Concepts
- What problem does the virtual DOM solve, and what are its limitations?
- How does `React.memo` differ from `useMemo`? When should you reach for each?
- Explain how the `useCallback` hook works and give a concrete example where omitting it causes a bug.
- What is the `key` prop actually doing internally? What happens if you use array index as a key?
- How does `useRef` differ from `useState`? Give two distinct use cases for `useRef`.
- What is a React portal and when would you use one?

## Hooks & State
- Explain the rules of hooks. Why do these restrictions exist?
- How does `useEffect` determine when to re-run? Describe the dependency array in detail.
- What is the difference between `useLayoutEffect` and `useEffect`? When does the timing matter?
- How would you implement a custom hook to debounce a search input?
- What is `useReducer` and when is it preferable over multiple `useState` calls?
- Explain `useContext` — what problem does it solve and what are its performance pitfalls?

## Performance Optimisation
- How do you identify and fix unnecessary re-renders in a React app? Which tools do you use?
- What is code splitting and how do you implement it with `React.lazy` and `Suspense`?
- Explain what happens when you call `setState` inside a loop. How does React batch updates?
- What is concurrent rendering in React 18? How does `useTransition` help with it?
- Describe a real performance problem you solved in a React project and your approach.

## State Management
- Compare local state, Context API, Redux Toolkit, and Zustand. When do you pick each?
- What is the "prop drilling" problem and what are the different ways to solve it?
- What is a selector in Redux? Why are they important for performance?
- Explain how React Query (TanStack Query) handles caching, stale-while-revalidate, and background refetching.
- How would you manage server state vs client-side UI state in a large application?

## Component Architecture
- What is component composition vs inheritance? Why does React favour composition?
- Explain the render props pattern and the higher-order component (HOC) pattern. What problems do they solve?
- What is the compound component pattern? Give an example where it improves an API.
- How would you design a reusable, accessible `<Modal>` component from scratch?
- What does "lifting state up" mean and when does it become a problem?

## TypeScript with React
- How do you type a component that accepts children with a specific shape?
- What is the difference between `React.FC` and a regular function signature for components?
- How do you type a `useRef` that will be assigned to a DOM element?
- Explain discriminated unions and give a React use case where they prevent bugs.

## Testing
- What is the difference between shallow rendering and full rendering in React Testing Library?
- How do you test a component that makes an API call? How do you mock it?
- Explain `findBy`, `getBy`, and `queryBy` queries — when do you use each?
- How do you test custom hooks in isolation using `renderHook`?
- What makes a test brittle? How does Testing Library's philosophy help avoid it?

## Accessibility & Best Practices
- What ARIA attributes are most important for a custom dropdown/select component?
- How do you manage keyboard focus in a modal dialog to meet WCAG standards?
- What is the difference between `role`, `aria-label`, and `aria-labelledby`?
- How does React's `StrictMode` help you catch bugs and prepare for concurrent mode?
