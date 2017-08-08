function getRandomLetter() {
  let randomIndex = Math.random()*(letters.length);
  return letters[randomIndex];
}

class Quiz {
  constructor(selector) {
    this.selector = selector;
    this.letters = ['B','C','D','F','G','H','I','K','L','O','P','Q','R','U','V','W','X','Y'];
    this.scores = [];
    for(let i = 0; i < 26; i++) {
      this.scores.push({
        score : 1,
        letter : String.fromCharCode(i + 'A'.charCodeAt(0))
      });
    }
    this.currentLetter = null;
    this.mode = 'train';

    this.testLetterDiv = this.selector.select('#testLetter');
    this.predictionDiv = this.selector.select('#prediction');
    this.modeDiv = this.selector.select('#modeDiv');
    this.imagePath = '/static/letters/';
    this.letterImageDiv = this.selector.select('#letterImage');
    this.trainScoreDiv = this.selector.select('#trainScore');
    this.testScoreDiv = this.selector.select('#testScore');
    this.height = 200;
    this.width = 600;
    this.drawScore();
    this.initTrainMode();
  }

  drawScore() {
    this.svg = this.testScoreDiv.append('svg').attr('height',this.height)
                                .attr('width',this.width);

    this.margin = {top:10,bottom:50,left:20,right:0};
    this.padding = {top:10,bottom:30,left:10,right:20};
    this.xScale = d3.scaleBand().rangeRound([this.padding.left, this.width-this.padding.right]).padding(0.1);
    this.yScale = d3.scaleLinear().rangeRound([this.height-this.padding.bottom, 0]);

    this.g = this.svg.append("g")
                .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")");
    this.xScale.domain(this.scores.map(function(d) { return d.letter; }));
    this.yScale.domain([0, 10]);
    this.g.append("g")
      .attr("class", "axis axis--x")
      .attr("transform", "translate(0," + (this.height-this.padding.bottom) + ")")
      .call(d3.axisBottom(this.xScale));

    this.g.append("g")
    .attr("class", "axis axis--y")
    .call(d3.axisLeft(this.yScale).ticks(10))
  .append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 4)
    .attr("dy", "0.71em")
    .attr("text-anchor","end")
    .style('fill','black')
    .text("Score");

    this.g.selectAll(".bar")
    .data(this.scores)
    .enter().append("rect")
    .attr("class", "bar")
    .attr("x", (d) => { return this.xScale(d.letter); })
    .attr("y", (d) => { return this.yScale(d.score); })
    .attr("width", this.xScale.bandwidth())
    .attr("height", (d) => { return this.height - this.yScale(d.score) - this.padding.bottom; });
  }

  updateScore() {
    this.g.selectAll('.bar')
      .data(this.scores)
      .transition(200)
      .attr("y", (d) => { return this.yScale(d.score); })
      .attr("height", (d) => { return this.height - this.yScale(d.score) - this.padding.bottom; });
  }

  initTrainMode() {
    this.mode = 'train';
    this.modeDiv.html('Training Mode');
    this.predictionDiv.html('No Prediction');
    this.letterImageDiv.style('display','block');
    this.trainScoreDiv.style('display','block');
    this.testScoreDiv.style('display','none');
    this.nextLetter();
  }

  initTestMode() {
    this.mode = 'test';
    this.modeDiv.html('Testing Mode');
    this.predictionDiv.html('No Prediction');
    this.letterImageDiv.style('display','none');
    this.trainScoreDiv.style('display','none');
    this.testScoreDiv.style('display','block');
    this.nextLetter();
  }

  showCorrect() {
    this.predictionDiv.html('Correct !');
  }

  showWrong() {
    this.predictionDiv.html('Wrong !');
  }

  switchMode() {
    if(this.mode == 'train') {
      this.initTestMode();
    }else {
      this.initTrainMode();
    }
  }

  nextLetter() {
    let randomIndex = Math.floor(Math.random()*(this.letters.length));
    this.currentLetter = this.letters[randomIndex];
    this.testLetterDiv.html(this.currentLetter);
    this.letterImageDiv.property('src',`${this.imagePath}/${this.currentLetter}.jpg`);
  }

  predict(letter) {
    letter = letter.toUpperCase();
    console.log(letter);
    if(letter == this.currentLetter) {
      this.showCorrect();
      this.scores[letter.charCodeAt(0) - 'A'.charCodeAt(0)].score++;
      this.updateScore();
      this.nextLetter();
    } else {
      this.showWrong(letter);
      if(this.mode == 'test') {
        this.nextLetter();
      }
    }
  }
}
