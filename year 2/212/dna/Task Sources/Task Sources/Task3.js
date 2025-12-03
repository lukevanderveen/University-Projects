"use strict";

const testlib = require( './testlib.js' );
let test = [];
let Patterns = [];
let matches = [];
let count = [];

// MAP: key: sequence symbol, value: array of alternative sequence symbols 

const nucliotides = {
    'R': ['G', 'A'],
    'Y': ['T', 'C'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'S': ['G', 'C'],
    'W': ['A', 'T'],
    'B': ['G', 'T', 'C'],
    'D': ['A', 'C', 'G'],
    'H': ['A', 'C', 'T'],
    'V': ['G', 'C', 'A'],
    'N': ['A', 'G', 'T', 'C']
};

function genCombos(pattern) {
    if (pattern.length === 0) {
        return [""];
    }

    const current = pattern[0];
    const rest = pattern.slice(1);
    const possibleChar = nucliotides[current] || [current];
    const patternNoCurrent = genCombos(rest);

    const combos = possibleChar.flatMap(nucleus =>
        patternNoCurrent.map(combination => nucleus + combination)
    );

    return combos;
}

function genSequenceCombos(patterns) {
    return patterns.flatMap(pattern => genCombos(pattern));
}

function findMatches(test, patterns) {
    patterns.forEach((pattern, patternIndex) => {
        const patternCombos = genSequenceCombos([pattern]);
        test.forEach((item, testIndex) => {
            if (match(pattern, item) || match(item, pattern)) {
                testlib.foundMatch(pattern, testIndex);
                matches.push({pattern});
            }
        }); 
    });

    return matches;
}

function match(pattern, item) {
    const regex = new RegExp(`(?:${pattern})`, 'g');
    return regex.test(item);
}




testlib.on( 'ready', function( patterns ) {
    console.log( "Patterns:", patterns );
    Patterns = patterns;
    testlib.runTests();
    Patterns.forEach(Pattern => {
        let combos = genSequenceCombos([Pattern]);
        console.log(`Pattern: ${Pattern}`);
        console.log('Sequence Combos:', combos);
    });
} );

testlib.on( 'data' , function ( data ) {
    test.push(data);

    let dna = test.join('');
    let regex = new RegExp(`(${Patterns.join('|')})`, 'g');
    let matches = dna.match(regex) || [];
    
    matches.forEach(match => {
        count[match] = (count[match] || 0) + 1;
    });
/*
    console.log("data:", data);
    console.log("Matches:", matches);
    console.log("Count:", count);
    console.log("dna:", dna);
    */
});


testlib.on( 'reset' , function() {
    console.log(test);
    Patterns.forEach(Pattern => {
        let combos = genSequenceCombos([Pattern]);
        matches = findMatches(test, combos);
    });

    test = [];
} );
   
testlib.on( 'end' , function() {
    testlib.frequencyTable(count);
});


testlib.setup( 3 ); // Runs test 1 (task1.data and task1.seq)

/*
I don't believe i got this completely working, as it seems to find matches at almost every index

*/