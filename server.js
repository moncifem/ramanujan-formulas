const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { evaluate } = require('mathjs');
const Decimal = require('decimal.js');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'client/build')));

// Mathematical constants with high precision
const CONSTANTS = {
  pi: new Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679'),
  e: new Decimal('2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274'),
  phi: new Decimal('1.6180339887498948482045868343656381177203091798057628621354486227052604628189024497072072041893911374'),
  gamma: new Decimal('0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495'),
  sqrt2: new Decimal('1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727'),
  sqrt3: new Decimal('1.7320508075688772935274463415058723669428052538103806280558069794519330169088000370811461867572485757'),
  ln2: new Decimal('0.6931471805599453094172321214581765680755001343602552541206800094933936219696947156058633269964186875')
};

// Evaluation engine
class MathEvaluator {
  constructor() {
    this.precision = 50; // decimal places
    Decimal.set({ precision: this.precision + 10 });
  }

  // Safely evaluate mathematical expression
  evaluateExpression(expression, constants = {}) {
    try {
      // Replace constants in expression
      let expr = expression.toLowerCase();
      
      // Replace mathematical constants
      expr = expr.replace(/\bpi\b/g, CONSTANTS.pi.toString());
      expr = expr.replace(/\be\b/g, CONSTANTS.e.toString());
      expr = expr.replace(/\bphi\b/g, CONSTANTS.phi.toString());
      expr = expr.replace(/\bgamma\b/g, CONSTANTS.gamma.toString());
      expr = expr.replace(/\bsqrt2\b/g, CONSTANTS.sqrt2.toString());
      expr = expr.replace(/\bsqrt3\b/g, CONSTANTS.sqrt3.toString());
      expr = expr.replace(/\bln2\b/g, CONSTANTS.ln2.toString());
      
      // Replace custom constants
      Object.keys(constants).forEach(key => {
        const regex = new RegExp(`\\b${key}\\b`, 'g');
        expr = expr.replace(regex, constants[key].toString());
      });

      // Use mathjs for evaluation with high precision
      const result = evaluate(expr);
      return new Decimal(result.toString());
      
    } catch (error) {
      throw new Error(`Expression evaluation failed: ${error.message}`);
    }
  }

  // Calculate approximation quality
  calculateApproximationQuality(computed, target, expression) {
    try {
      const computedDecimal = new Decimal(computed.toString());
      const targetDecimal = new Decimal(target.toString());
      
      // Calculate absolute error
      const absoluteError = computedDecimal.minus(targetDecimal).abs();
      
      // Calculate relative error
      const relativeError = absoluteError.dividedBy(targetDecimal.abs());
      
      // Calculate complexity score (based on expression length and operators)
      const complexity = this.calculateComplexity(expression);
      
      // Calculate elegance score (lower is better)
      const eleganceScore = absoluteError.times(new Decimal(1).plus(new Decimal(0.01).times(complexity)));
      
      // Calculate accuracy (number of correct decimal places)
      const accuracy = this.calculateAccuracy(absoluteError);
      
      return {
        absoluteError: absoluteError.toString(),
        relativeError: relativeError.toString(),
        complexity,
        eleganceScore: eleganceScore.toString(),
        accuracy,
        score: this.calculateOverallScore(absoluteError, complexity)
      };
      
    } catch (error) {
      throw new Error(`Quality calculation failed: ${error.message}`);
    }
  }

  calculateComplexity(expression) {
    // Count various elements that contribute to complexity
    const operators = (expression.match(/[+\-*/^()]/g) || []).length;
    const functions = (expression.match(/\b(sqrt|log|ln|exp|sin|cos|tan|asin|acos|atan)\b/g) || []).length;
    const constants = (expression.match(/\b(pi|e|phi|gamma|sqrt2|sqrt3|ln2)\b/g) || []).length;
    const numbers = (expression.match(/\d+\.?\d*/g) || []).length;
    
    return expression.length + operators * 2 + functions * 3 + constants + numbers;
  }

  calculateAccuracy(absoluteError) {
    if (absoluteError.equals(0)) return this.precision;
    
    const logError = absoluteError.log();
    const accuracy = logError.dividedBy(new Decimal(10).log()).negated().floor();
    
    return Math.max(0, Math.min(this.precision, accuracy.toNumber()));
  }

  calculateOverallScore(absoluteError, complexity) {
    // Higher score is better
    const errorScore = new Decimal(1).dividedBy(absoluteError.plus(new Decimal('1e-50')));
    const complexityPenalty = new Decimal(1).dividedBy(new Decimal(complexity).plus(1));
    
    return errorScore.times(complexityPenalty).toNumber();
  }
}

const evaluator = new MathEvaluator();

// API Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'Ramajan Mathematical Evaluator is running',
    timestamp: new Date().toISOString()
  });
});

// Get available constants
app.get('/api/constants', (req, res) => {
  const constantsInfo = Object.keys(CONSTANTS).map(key => ({
    name: key,
    value: CONSTANTS[key].toString(),
    description: getConstantDescription(key)
  }));
  
  res.json({ constants: constantsInfo });
});

// Evaluate single approximation
app.post('/api/evaluate', (req, res) => {
  try {
    const { expression, target, targetName } = req.body;
    
    if (!expression || !target) {
      return res.status(400).json({ 
        error: 'Missing required fields: expression and target' 
      });
    }

    // Evaluate the expression
    const computed = evaluator.evaluateExpression(expression);
    
    // Calculate quality metrics
    const quality = evaluator.calculateApproximationQuality(computed, target, expression);
    
    const result = {
      expression,
      target: target.toString(),
      targetName: targetName || 'unknown',
      computed: computed.toString(),
      quality,
      timestamp: new Date().toISOString()
    };
    
    res.json({ success: true, result });
    
  } catch (error) {
    res.status(500).json({ 
      error: error.message,
      success: false 
    });
  }
});

// Evaluate batch of approximations
app.post('/api/evaluate-batch', (req, res) => {
  try {
    const { approximations } = req.body;
    
    if (!Array.isArray(approximations)) {
      return res.status(400).json({ 
        error: 'approximations must be an array' 
      });
    }

    const results = [];
    const errors = [];

    approximations.forEach((approx, index) => {
      try {
        const { expression, target, targetName, description } = approx;
        
        if (!expression || !target) {
          errors.push({
            index,
            error: 'Missing required fields: expression and target'
          });
          return;
        }

        // Evaluate the expression
        const computed = evaluator.evaluateExpression(expression);
        
        // Calculate quality metrics
        const quality = evaluator.calculateApproximationQuality(computed, target, expression);
        
        results.push({
          index,
          expression,
          target: target.toString(),
          targetName: targetName || 'unknown',
          description: description || '',
          computed: computed.toString(),
          quality,
          timestamp: new Date().toISOString()
        });
        
      } catch (error) {
        errors.push({
          index,
          expression: approx.expression || 'unknown',
          error: error.message
        });
      }
    });

    // Sort results by score (best first)
    results.sort((a, b) => b.quality.score - a.quality.score);

    res.json({ 
      success: true, 
      results,
      errors,
      summary: {
        total: approximations.length,
        successful: results.length,
        failed: errors.length,
        bestScore: results.length > 0 ? results[0].quality.score : 0
      }
    });
    
  } catch (error) {
    res.status(500).json({ 
      error: error.message,
      success: false 
    });
  }
});

// Get example approximations
app.get('/api/examples', (req, res) => {
  const examples = [
    {
      expression: "22/7",
      target: CONSTANTS.pi.toString(),
      targetName: "π",
      description: "Classic rational approximation to π"
    },
    {
      expression: "355/113",
      target: CONSTANTS.pi.toString(),
      targetName: "π",
      description: "Milü's approximation to π"
    },
    {
      expression: "(1 + sqrt(5))/2",
      target: CONSTANTS.phi.toString(),
      targetName: "φ",
      description: "Golden ratio exact formula"
    },
    {
      expression: "sqrt(2 + sqrt(2 + sqrt(2)))",
      target: "1.847759065022573512256366378793576573644833252847264",
      targetName: "nested radical",
      description: "Nested radical approximation"
    },
    {
      expression: "exp(pi * sqrt(163))",
      target: "262537412640768744",
      targetName: "Ramanujan constant",
      description: "Famous near-integer discovered by Ramanujan"
    }
  ];
  
  res.json({ examples });
});

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'client/build', 'index.html'));
});

// Helper function
function getConstantDescription(name) {
  const descriptions = {
    pi: "π - Ratio of circumference to diameter",
    e: "e - Euler's number, base of natural logarithm",
    phi: "φ - Golden ratio, (1+√5)/2",
    gamma: "γ - Euler-Mascheroni constant",
    sqrt2: "√2 - Square root of 2",
    sqrt3: "√3 - Square root of 3",
    ln2: "ln(2) - Natural logarithm of 2"
  };
  return descriptions[name] || "Mathematical constant";
}

// Start server
app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██████╗  █████╗ ███╗   ███╗ █████╗      ██╗ █████╗ ███╗   ██╗           ║
║    ██╔══██╗██╔══██╗████╗ ████║██╔══██╗     ██║██╔══██╗████╗  ██║           ║
║    ██████╔╝███████║██╔████╔██║███████║     ██║███████║██╔██╗ ██║           ║
║    ██╔══██╗██╔══██║██║╚██╔╝██║██╔══██║██   ██║██╔══██║██║╚██╗██║           ║
║    ██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║╚█████╔╝██║  ██║██║ ╚████║           ║
║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝           ║
║                                                                              ║
║    Mathematical Approximation Evaluator                                     ║
║    Server running on port ${PORT}                                               ║
║    ASCII vibes activated ⚡                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
  `);
});

module.exports = app;
