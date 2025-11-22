import java.io.File
import kotlin.random.Random

fun isValid(word: String): Boolean {
    if (word.length == 5) {
        return (true)
    } else {
        return (false)
    }

    // Returns true if the given word is valid in Wordle
    // (i.e., if it consists of exactly 5 letters)
}


fun readWordList(filename: String): MutableList<String> {
   val words = mutableListOf<String>()

    File(filename).forEachLine { line ->
        val word = line.trim().lowercase()
        if (isValid(word)) {
            words.add(word)
        }
    }

    return words
    // Reads Wordle target words from the specified file, returning them as a list of strings.
    // (See the Baeldung article on reading from files in Kotlin for help with this.)
}


fun pickRandomWord(words: MutableList<String>): String {
    val wordNum = Random.nextInt(words.size)
    return words.removeAt(wordNum)
    // Chooses a random word from the given list, removes that word from the list, then
    // returns it.
}


fun obtainGuess(attempt: Int): String {
    while (true) {
        println("Attempt $attempt: ")
        val guess = readln()

        if (isValid(guess)) {
            return guess
        } else {
            println("Invalid word, try again.")
        }
    }
    // Prints a prompt using the given attempt number (e.g. "Attempt 1: "), then reads a word
    // from stdin. The word should be returned if valid, otherwise the user should be prompted
    // to try again.
}


fun evaluateGuess(guess: String, target: String): List<Int> {
    val results = mutableListOf<Int>()
    
    for (i in 0 until 5) {
        if (guess[i] == target[i]) {
            results.add(1)
        } else {
            results.add(0)
        }
    }

    return results
    // Compares a guess with the target word. Returns a list containing 5 integers,
    // representing the result of comparison at each letter position. 0 indicates no match, 1
    // indicates a match.
}


fun displayGuess(guess: String, matches: List<Int>) {
    for (i in 0 until 5) {
        if (matches[i] == 1) {
            print(guess[i])
        } else {
            print("?")
        }
    }
    println()
    // Displays the letters of a guess that match target word, or a ‘?’ character where there is
    // no match.
}