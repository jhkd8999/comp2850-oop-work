// Task 5.1.1: anagram checking using a function
fun anagrams(first: String, second: String): Boolean {
    if (first.length != second.length) {
        return false
    }
    val firstChars = first.lowercase().toList().sorted()
    val secondChars = second.lowercase().toList().sorted()
    return firstChars == secondChars
}

fun main() {
    println("Enter string 1: ")
    val string1 = readln()
    println("Enter string 2: ")
    val string2 = readln()

    println(anagrams(string1, string2))

}