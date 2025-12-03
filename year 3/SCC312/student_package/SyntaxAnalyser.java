import java.io.IOException;

public class SyntaxAnalyser extends AbstractSyntaxAnalyser{
    String programName;

    public SyntaxAnalyser(String filename) throws IOException{
        this.lex = new LexicalAnalyser(filename); // Initialize lexical analyser
        this.programName = filename;
    }

    // deubg assistant for reporting errors in terminal, unused 
    private void handleParseError(String context, CompilationException e) throws CompilationException {
        String newMessage = String.format(
            "Syntax Error in program '%s': In %s -> %s\nStack Trace:\n%s",
            programName, context, e.getMessage(), e.toTraceString()  // Add full stack trace
        );

        System.err.println(newMessage); // ✅ Print error before throwing
        //throw new CompilationException(newMessage); causes infinite loop
    }
    
    
    
    public void acceptTerminal(int symbol) throws IOException, CompilationException {
        try{
            if (nextToken.symbol == symbol) {
                myGenerate.insertTerminal(nextToken); // Log matched token
                nextToken = lex.getNextToken(); // Move to the next token
            } else {
                String foundToken = nextToken.text.isEmpty() ? "EOF" : "'" + nextToken.text + "' (" + Token.getName(nextToken.symbol) + ")";
                String message = String.format(
                    "Syntax Error in program '%s' at line %d: Expected %s but found %s",
                    programName, nextToken.lineNumber, Token.getName(symbol), foundToken);
                myGenerate.reportError(nextToken, message);
                //throw new CompilationException(message);  // commenting this out removes stack trace
            }
        } catch(CompilationException e){
            //handleParseError("accept terminal " +Token.getName(symbol), e);
            throw e;
        }
        
    }

    // Grammar rules section
    // for the majority i encapsulate each grammar rule in a try catch
    // so that if there are errors preventing going into any conditions
    // compilation errors will be caught

    /** Begin processing the first (top level) token.*/
    public void _statementPart_() throws IOException, CompilationException{
        try{
            myGenerate.commenceNonterminal("StatementPart");
            myGenerate.insertTerminal(nextToken);
            if (nextToken.symbol == Token.beginSymbol) {
                nextToken = lex.getNextToken();
                _statementList_();    
                acceptTerminal(Token.endSymbol);   
            } else {
                String message = String.format("Syntax Error in program '%s' at line %d: Expected 'begin' at the start of StatementPart, but found '%s'", programName, nextToken.lineNumber, nextToken.text);
                myGenerate.reportError(nextToken, message);
                //throw new CompilationException(message); 

            }
            myGenerate.finishNonterminal("StatementPart");
        }catch (CompilationException e){
            //handleParseError("StatementPart ", e);
            
        }
        
    }

    //grammar rules for statement list, calls statemenlist recursively if 
    //semi colons are present
    private void _statementList_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("StatementList");

            _statement_();

            while (nextToken.symbol == Token.semicolonSymbol) { // If ';' found, continue
                acceptTerminal(Token.semicolonSymbol);
                //System.out.println("statement list next token = "+ Token.getName(nextToken.symbol));
                //nextToken = lex.getNextToken();
                _statementList_();
            }
            myGenerate.finishNonterminal("StatementList");
        }catch (CompilationException e){
            //handleParseError("StatementList ", e);
            throw e;
        }
        
    }
    
    // grammar rules for statement, switch case based on next token symbol 
    // for different statement types
    public void _statement_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("Statement");

            //System.out.println("statement next token = "+ Token.getName(nextToken.symbol));
            switch (nextToken.symbol){
                case Token.identifier:
                    _assignmentStatement_();
                    break;
                case Token.ifSymbol:
                    _ifStatement_();
                    break;
                case Token.whileSymbol:
                    _whileStatement_();
                    break;
                case Token.callSymbol:
                    _procedureStatement_();
                    break;
                case Token.doSymbol:
                    _untilStatement_();
                    break;
                case Token.forSymbol:
                    _forStatement_();
                    break;
                default:
                    String message = String.format("Syntax Error in program '%s' at line %d: Unexpected token '%s' in Statement", programName, nextToken.lineNumber, nextToken.text);
                    myGenerate.reportError(nextToken, message);

                    //throw new CompilationException(message); 
            }
            myGenerate.finishNonterminal("Statement");
        }catch (CompilationException e){
            //handleParseError("Statement ", e);
            throw e;
        }
        
    }
      
    // assignment statement,selection for identifier, number constatnt or string constant
    // else report an error
    private void _assignmentStatement_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("AssignmentStatement");
            //System.out.println("statement next token = "+ Token.getName(nextToken.symbol));
            //nextToken = lex.getNextToken();

            acceptTerminal(Token.identifier);
            acceptTerminal(Token.becomesSymbol);

            if (nextToken.symbol == Token.identifier || nextToken.symbol == Token.numberConstant) {
                _expression_();
            } else if (nextToken.symbol == Token.stringConstant) {
                acceptTerminal(Token.stringConstant);
            } else {
                String message = String.format("Syntax Error in program '%s' at line %d: Unexpected token '%s' in AssignmentStatement", programName, nextToken.lineNumber, nextToken.text);
                myGenerate.reportError(nextToken, message);
                //throw new CompilationException(message); 
            }

            myGenerate.finishNonterminal("AssignmentStatement");
        }catch (CompilationException e){
            //handleParseError("AssignmentStatement ", e);
            throw e;
        }
        
    }
    
    // if statement grammar rule 
    private void _ifStatement_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("IfStatement");
            acceptTerminal(Token.ifSymbol);
            _condition_();
            acceptTerminal(Token.thenSymbol);
            _statementList_();

            if (nextToken.symbol == Token.elseSymbol) {
                acceptTerminal(Token.elseSymbol);
                _statementList_();
            }

            acceptTerminal(Token.endSymbol);
            acceptTerminal(Token.ifSymbol);
            myGenerate.finishNonterminal("IfStatement");
        }catch (CompilationException e){
            //handleParseError("IfStatement ", e);
            throw e;
        }
        
    }

    // while statement grammar rule
    private void _whileStatement_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("WhileStatement");
            acceptTerminal(Token.whileSymbol);
            _condition_();
            acceptTerminal(Token.loopSymbol);
            _statementList_();
            acceptTerminal(Token.endSymbol);
            acceptTerminal(Token.loopSymbol);
            myGenerate.finishNonterminal("WhileStatement");
        }catch (CompilationException e){
            //handleParseError("WhileStatement ", e);
            throw e;
        }
    }

    // procedure statement grammar rule
    private void _procedureStatement_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("ProcedureStatement");
            acceptTerminal(Token.callSymbol);
            acceptTerminal(Token.identifier);
            acceptTerminal(Token.leftParenthesis);
            _argumentList_();
            acceptTerminal(Token.rightParenthesis);
            myGenerate.finishNonterminal("ProcedureStatement");
        }catch (CompilationException e){
            //handleParseError("ProcedureStatement ", e);
            throw e;
        }
    }

    // until statement grammar rule
    private void _untilStatement_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("UntilStatement");
            acceptTerminal(Token.doSymbol);
            _statementList_();
            acceptTerminal(Token.untilSymbol);
            _condition_();
            myGenerate.finishNonterminal("UntilStatement");
        }catch (CompilationException e){
            //handleParseError("UntilStatement ", e);
            throw e;
        }
    }

    // grammar rule for for statement
    private void _forStatement_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("ForStatement");
            acceptTerminal(Token.forSymbol);
            acceptTerminal(Token.leftParenthesis); // creates this '(' (()
            _assignmentStatement_();
            acceptTerminal(Token.semicolonSymbol);
            _condition_();
            acceptTerminal(Token.semicolonSymbol);
            _assignmentStatement_();
            acceptTerminal(Token.rightParenthesis); // creates: ')' ())
            acceptTerminal(Token.doSymbol);
            _statementList_();
            acceptTerminal(Token.endSymbol);
            acceptTerminal(Token.loopSymbol);
            myGenerate.finishNonterminal("ForStatement");
        }catch (CompilationException e){
            //handleParseError("ForStatement ", e);
            throw e;
        }
    }

    // grammar rule for argument list, works like statement list,
    // recursively call argument list if commas persist
    private void _argumentList_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("ArgumentList");
            acceptTerminal(Token.identifier);

            while (nextToken.symbol == Token.commaSymbol) { // If ';' found, continue
                acceptTerminal(Token.commaSymbol);
                _argumentList_();
                acceptTerminal(Token.identifier);
            }
            myGenerate.finishNonterminal("ArgumentList");
        }catch (CompilationException e){
            //handleParseError("ArgumentList ", e);
            throw e;
        }
        
    }

    // condition grammar rule, use an if statement ORing the different
    // conditions, else generate error
    private void _condition_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("Condition");

            acceptTerminal(Token.identifier); // First operand
            _conditionalOperator_(); // Accept conditional operator
        
            // Accept second operand (identifier, number, or string constant)
            if (nextToken.symbol == Token.identifier || nextToken.symbol == Token.numberConstant || nextToken.symbol == Token.stringConstant) {
                acceptTerminal(nextToken.symbol);
            } else {
                String message = String.format("Syntax Error in program '%s' at line %d: Unexpected Condition token '%s'", programName, nextToken.lineNumber, nextToken.text);
                myGenerate.reportError(nextToken, message);
                //throw new CompilationException(message); 
            }
        
            myGenerate.finishNonterminal("Condition");
        }catch (CompilationException e){
            //handleParseError("Condition ", e);
            throw e;
        }
        
    }

    // conditional operator grammar rule, uses a if statement ORing the different
    // operator types
    private void _conditionalOperator_() throws IOException, CompilationException {
        try {
            // add the commence terminal etc etc
            myGenerate.commenceNonterminal("ConditionalOperator");
            if (nextToken.symbol == Token.greaterThanSymbol || nextToken.symbol == Token.greaterEqualSymbol ||
                nextToken.symbol == Token.equalSymbol || nextToken.symbol == Token.notEqualSymbol ||
                nextToken.symbol == Token.lessThanSymbol || nextToken.symbol == Token.lessEqualSymbol) {
                acceptTerminal(nextToken.symbol);
            } else {
                String message = String.format("Syntax Error in program '%s' at line %d: Unexpected ConditionalOperator token '%s'", programName, nextToken.lineNumber, nextToken.text);
                myGenerate.reportError(nextToken, message);
                //throw new CompilationException(message); 
            }
            myGenerate.finishNonterminal("Expression");
        }catch (CompilationException e){
            //handleParseError("ConditionalOperator ", e);
            throw e;
        }
        
    }

    // expression grammar rule, calls term repeatedly throughout the expression unil
    // there are no more specified operators remaining 
    private void _expression_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("Expression");

            _term_(); // Parse first term

            while (nextToken.symbol == Token.plusSymbol || nextToken.symbol == Token.minusSymbol) { // Handle `+` or `-`
                acceptTerminal(nextToken.symbol);
                _term_();
            }

            myGenerate.finishNonterminal("Expression");
        }catch (CompilationException e){
            //handleParseError("Expression ", e);
            throw e;
        }
        
    }

    // term grammar rule, calls factor repeatedly throughout the expression unil
    // there are no more specified operators remaining 
    private void _term_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("Term");

            _factor_(); // Parse first factor

            while (nextToken.symbol == Token.timesSymbol || nextToken.symbol == Token.divideSymbol || nextToken.symbol == Token.modSymbol) { // Handle `*`, `/`, `%`
                acceptTerminal(nextToken.symbol);
                _factor_();
            }

            myGenerate.finishNonterminal("Term");
        }catch (CompilationException e){
            //handleParseError("Term ", e);
            throw e;
        }
       
    }

    // factor grammar rule, selects terminal to accept based on specificed next token
    // otherwise generates and error report
    private void _factor_() throws IOException, CompilationException{
        try {
            myGenerate.commenceNonterminal("Factor");

            if (nextToken.symbol == Token.identifier) {
                acceptTerminal(Token.identifier);
            } else if (nextToken.symbol == Token.numberConstant) {
                acceptTerminal(Token.numberConstant);
            } else if (nextToken.symbol == Token.leftParenthesis) {
                acceptTerminal(Token.leftParenthesis);
                _expression_();
                acceptTerminal(Token.rightParenthesis);
            } else {
                String message = String.format("Syntax Error in program '%s' at line %d: Unexpected Factor token '%s'", programName, nextToken.lineNumber, nextToken.text);
                myGenerate.reportError(nextToken, message);
                //throw new CompilationException(message); 
            }

            myGenerate.finishNonterminal("Factor");
        }catch (CompilationException e){
            //handleParseError("Factor", e);
            throw e;
        }  
    }
}
