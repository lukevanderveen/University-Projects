public class Generate extends AbstractGenerate{

    
    public void reportError(Token token, String explanatoryMessage) throws CompilationException {
        // Print detailed error information
        //System.err.println("Compilation Error");
        //System.err.println("report Error here");


        /**System.err.println("Message: " + explanatoryMessage);
        if (explanatoryMessage.contains("Expected if")) {
            System.err.println("Hint: 'if' statements must be followed by a condition in parentheses.");
        } else if (explanatoryMessage.contains("Expected loop")) {
            System.err.println("Hint: Ensure 'while' statements end with 'loop' before the body.");
        } else if (explanatoryMessage.contains("Unexpected")) {
            System.err.println("Hint: Unexpected tokens may indicate a missing or misplaced statement.");
        }
        System.err.println();**/
        throw new CompilationException(explanatoryMessage);
        // Throw the exception as required by spec
    }
    
}