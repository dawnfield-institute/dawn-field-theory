# Entropy Logic Report

**Input Sentence:** Before you leave for lunch, make sure all systems are shut down and the logs are archived.

## Instruction Trace
- DO 'Before you leave for lunch, make sure all systems are shut down and the logs are archived.'
-   IF 'Before' THEN
-   DO 'you leave for lunch, make sure all systems are shut down and the logs are archived.'
-     END 'you'
-     JUNCTION 'leave for lunch, make sure all systems are shut down and the logs are archived.'
-       END 'leave'
-       DO 'for lunch, make sure all systems are shut down and the logs are archived.'
-         END 'for'
-         IF 'lunch, make sure all systems are shut down and the logs are archived.' THEN

## Top Node Summary
- Role: driver
- Entropy: 4.056
- Text: `Before you leave for lunch, make sure all systems are shut down and the logs are archived.`
