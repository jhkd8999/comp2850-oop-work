fun main() {
    println("PIZZA MENU")
    println("(a) Margherita")
    println("(b) Quattro Stagioni")
    println("(c) Seafood")
    println("(d) Hawaiian")

    println("Choose your pizza (a-d): ")
    val choice = readln().lowercase()

    if (choice.length == 1 && choice[0] in 'a'..'d') {
        println("Order accepted")
    }
    else {
        println("Invalid choice!")
    }
}