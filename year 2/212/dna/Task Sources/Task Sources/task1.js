"use strict";

const testlib = require( './testlib.js' );
let test = [];
let matches = [];
let dna;
let count = {};



testlib.on( 'ready', function( patterns ) {
    console.log( "Patterns:", patterns );
    testlib.runTests();
} );

testlib.on( 'data', function( data ) {
    test.push(data);
    dna = test.join('');
    matches = dna.match(/(AA|CC|TT|GG)/g) || [];

} );


testlib.on( 'reset', function() {
    console.log(test);
    matches.forEach(match => {
        count[match] = (count[match] || 0)+ 1;
    });
    test = [];
} );
   
testlib.on('end', function() {
    console.log(count);
    testlib.frequencyTable(count);
});


testlib.setup( 1 ); // Runs test 1 (task1.data and task1.seq)

/*
blank space
>>




*/