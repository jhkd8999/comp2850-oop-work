// Task 5.4.1: string extension function
val String.tooLong: Boolean get() = this.length > 20

fun main() {
    println("Enter string: ")
    val input = readln()
    println(input.tooLong)
}