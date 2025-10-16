// Task 4.5: summing odd integers with a for loop
fun main() {
    println("Enter the limit: ")
    var limit = readln().toLong()

    var total = 0.toLong()
    for (n in 1.toLong()..limit step 2) {
        total = total + n
    }
    println("The sum of all numbers from 1 to ${limit} is ${total}")

}