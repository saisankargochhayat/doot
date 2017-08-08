function getRandomLetter() {
  let letters = ['B','C','D','F','G','H','I','K','L','O','P','Q','R','U','V','W','X','Y'];
  let randomIndex = Math.random()*(letters.length);
  return letters[randomIndex];
}
