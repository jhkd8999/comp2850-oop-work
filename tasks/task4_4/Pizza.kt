fun main() {
    println("PIZZA MENU")
    println("(a) Margherita")
    println("(b) Quattro Stagioni")
    println("(c) Seafood")
    println("(d) Hawaiian")

    var choice = ""
    while (choice.length != 1 || choice[0] !in 'a'..'d') {
        println("Choose your pizza (a-d): ")
        choice = readln().lowercase()
    }
    println("Order accepted")
}