const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { evaluate } = require('mathjs');
const Decimal = require('decimal.js');
const path = require('path');
require('dotenv').config();

// Import parser (can be edited without restart using nodemon)
const { parseBatch } = require('./parser.js');

const app = express();
const PORT = process.env.PORT || 5000;

// OpenRouter API configuration
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions';

if (!OPENROUTER_API_KEY) {
    console.warn('⚠️  OPENROUTER_API_KEY not found in environment variables');
    console.log('AI evaluation features will be disabled. Add OPENROUTER_API_KEY to .env file.');
}

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
      // Clean Python/mpmath syntax
      let expr = expression.toLowerCase()
        .replace(/mp\.pi/g, 'pi')
        .replace(/mp\.e/g, 'e')
        .replace(/mp\.sqrt\(/g, 'sqrt(')
        .replace(/mp\.exp\(/g, 'exp(')
        .replace(/mp\.ln\(/g, 'log(')
        .replace(/mp\.log\(/g, 'log(')
        .replace(/mp\.sin\(/g, 'sin(')
        .replace(/mp\.cos\(/g, 'cos(')
        .replace(/mp\.tan\(/g, 'tan(')
        .replace(/mp\.sinh\(/g, 'sinh(')
        .replace(/mp\.cosh\(/g, 'cosh(')
        .replace(/mp\.tanh\(/g, 'tanh(')
        .replace(/mp\./g, ''); // Remove any remaining mp. prefixes

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

// AI Evaluator using OpenRouter
class AIEvaluator {
    constructor() {
        this.systemPrompt = `You are a professional mathematician and numerical analyst. Your task is to rigorously evaluate mathematical approximations provided by an AI discovery system.

You must analyze each approximation based on these criteria:
1. ACCURACY: How close is the approximation to the true value? Include error analysis.
2. EFFICIENCY: How computationally efficient is the expression compared to alternatives?
3. NOVELTY: Is this approach TRULY NEW and NEVER PUBLISHED BEFORE?
   
   ⚠️ CRITICAL NOVELTY RULES - You must be EXTREMELY CONSERVATIVE:
   
   ALWAYS mark as "No" if the expression:
   - Is a classical formula (Ramanujan, Euler, Gauss, etc.)
   - Appears in OEIS, Wolfram MathWorld, Wikipedia, or any math database
   - Is a standard textbook identity or trivial manipulation
   - Uses well-known techniques (continued fractions, nested radicals, Taylor series)
   - Is similar to published work (even with different parameters)
   - Can be derived from known formulas through simple algebra
   - Is an obvious identity or special case
   
   ONLY mark as "Yes" if you are 300% CERTAIN that:
   - This EXACT form has NEVER appeared in any mathematical publication
   - It represents a genuinely new mathematical discovery
   - It would be publishable as a novel result in a peer-reviewed journal
   - You have thoroughly checked and found NO similar formulas in literature
   
   DEFAULT TO "No" IN ALL CASES OF UNCERTAINTY.

4. STABILITY: Does it converge reliably for intended inputs?
5. GENERALITY: What are the domain limitations and input ranges?

For each approximation, you must respond with ONLY a valid JSON object in this exact format:
{
  "accuracy": "High/Medium/Low + detailed error analysis",
  "efficiency": "High/Medium/Low + computational complexity notes",
  "novelty": "No + specific references to where this is known from",
  "stability": "Stable/Unstable + convergence analysis",
  "generality": "General/Restricted + domain limitations",
  "recommendation": "Use/Needs improvement/Discard"
}

Be precise, technical, and provide specific mathematical insights. Focus on practical applicability and mathematical rigor.`;
    }

    async evaluateApproximation(approximation, model = 'anthropic/claude-3.5-sonnet') {
        if (!OPENROUTER_API_KEY) {
            return {
                accuracy: "Unable to analyze - OpenRouter API key not configured",
                efficiency: "Unable to analyze - OpenRouter API key not configured",
                novelty: "Unable to analyze - OpenRouter API key not configured",
                stability: "Unable to analyze - OpenRouter API key not configured",
                generality: "Unable to analyze - OpenRouter API key not configured",
                recommendation: "Needs improvement"
            };
        }

        const userPrompt = `Evaluate this mathematical approximation:

Expression: ${approximation.expression}
Computed Value: ${approximation.computed}
Target Value: ${approximation.target}
Absolute Error: ${approximation.quality?.absoluteError || 'N/A'}
Relative Error: ${approximation.quality?.relativeError || 'N/A'}
Complexity: ${approximation.quality?.complexity || 'N/A'}
Accuracy: ${approximation.quality?.accuracy || 'N/A'} digits

Provide a rigorous mathematical evaluation focusing on accuracy, efficiency, novelty, stability, and generality. Respond with only the JSON evaluation object.`;

        try {
            const response = await fetch(OPENROUTER_API_URL, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://ramajan-evaluator.local',
                    'X-Title': 'Ramanujan Mathematical Evaluator'
                },
                body: JSON.stringify({
                    model: model,
                    messages: [
                        {
                            role: 'user',
                            content: this.systemPrompt + '\n\n' + userPrompt
                        }
                    ],
                    temperature: 0.1,
                    max_tokens: 2000
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`OpenRouter API error: ${response.status} - ${errorData.error?.message || 'Unknown error'}`);
            }

            const data = await response.json();
            const content = data.choices[0]?.message?.content;

            if (!content) {
                throw new Error('No content received from OpenRouter API');
            }

            try {
                // Try to extract JSON from markdown code blocks if present
                let jsonContent = content;
                const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
                if (jsonMatch) {
                    jsonContent = jsonMatch[1];
                } else {
                    // Try to find JSON object in the response
                    const objectMatch = content.match(/\{[\s\S]*\}/);
                    if (objectMatch) {
                        jsonContent = objectMatch[0];
                    }
                }
                
                const evaluation = JSON.parse(jsonContent);
                return evaluation;
            } catch (parseError) {
                console.error('Failed to parse AI response as JSON:', content);
                console.error('Parse error:', parseError.message);
                return {
                    accuracy: "AI response parsing failed - unable to analyze",
                    efficiency: "AI response parsing failed - unable to analyze", 
                    novelty: "AI response parsing failed - unable to analyze",
                    stability: "AI response parsing failed - unable to analyze",
                    generality: "AI response parsing failed - unable to analyze",
                    recommendation: "Needs improvement"
                };
            }

        } catch (error) {
            console.error('Error evaluating approximation:', error);
            return {
                accuracy: `Error during evaluation: ${error.message}`,
                efficiency: "Unable to analyze due to API error",
                novelty: "Unable to analyze due to API error", 
                stability: "Unable to analyze due to API error",
                generality: "Unable to analyze due to API error",
                recommendation: "Needs improvement"
            };
        }
    }
}

const aiEvaluator = new AIEvaluator();

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
    console.log('\n' + '='.repeat(80));
    console.log('📥 RECEIVED BATCH REQUEST');
    console.log('Request Body Keys:', Object.keys(req.body));
    
    const { approximations } = req.body;
    
    if (!approximations) {
      console.log('❌ ERROR: "approximations" field missing from request body');
      console.log('Full body sample:', JSON.stringify(req.body, null, 2).substring(0, 200));
      return res.status(400).json({ 
        error: 'Invalid request: approximations array required' 
      });
    }
    
    if (!Array.isArray(approximations)) {
      console.log('❌ ERROR: "approximations" is not an array');
      console.log('Type received:', typeof approximations);
      return res.status(400).json({ 
        error: 'approximations must be an array' 
      });
    }

    console.log(`✓ Received array with ${approximations.length} items`);
    
    // Parse approximations using external parser (hot-reloadable)
    delete require.cache[require.resolve('./parser.js')]; // Clear cache for hot reload
    const { parseBatch: parseBatchFresh } = require('./parser.js');
    const { parsed, errors: parseErrors } = parseBatchFresh(approximations);
    
    // Evaluate successfully parsed approximations
    const results = [];
    const errors = [...parseErrors];

    parsed.forEach((item) => {
      try {
        const { index, expression, targetValue, targetName, description } = item;

        // Evaluate the expression
        const computed = evaluator.evaluateExpression(expression);
        
        // Calculate quality metrics
        const quality = evaluator.calculateApproximationQuality(computed, targetValue, expression);
        
        results.push({
          index,
          expression,
          target: targetValue,
          targetName,
          description,
          computed: computed.toString(),
          quality,
          timestamp: new Date().toISOString()
        });
        
      } catch (error) {
        console.error(`Error evaluating approximation ${item.index}:`, error.message);
        errors.push({
          index: item.index,
          expression: item.expression || 'unknown',
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
    },
    {
      expression: "exp(pi * sqrt(522)) / (sinh(pi * sqrt(522)) / 32)",
      target: "64.000000000000000000000000000000000000000000000000000000000000289397998242896671259361635333225519137512354795290965495819234757973776223252645023028179314378013094795610564233390776147464537129009527336578321078928927792897119872167772819761467536980161285156101039695924499304568768360412554309046437971142599441246541624151359772829713097161325419085629734071097024747038578630582245234329568918722774238485132867093924306525170602828204598881922253955301385127224637809119763784330598738277680675709720955462109961307346988560056042973694169448154438019778714859670269611974151691007394151625051487438405456274601925673569345417146979546601957227615052706522343546730320780327037071215596638780809071531481722849232563728183072164413220018438998950218321001615416738266610279756138384791551079664301901177400804166363118616672245917011231443180498380589788685889297057705568592719198798101372318931511057513593973934989010941031141386993031267390882812095749671438774093831031030103860692266211461729491698362312985175940678988627102236112240354302837888087095990701401337144822183769687363895356725600116940243312170328850173794090300274157172864160588034379484752897902903923731612430241176108573904189121144354620414840606497389009034456210547896501410860410316586157993498287175351729082825157979822267710856308869476348081100417672238669162999975555544398620306643191297087191130454177771110201472172930764660073639324741510320492922156223298904299611214414853387014737911260326130536277368",
      targetName: "64",
      description: "Novel approximation to 64 using exponential and hyperbolic functions"
    }
  ];
  
  res.json({ examples });
});

// AI Evaluation endpoint
app.post('/api/ai-evaluate', async (req, res) => {
  try {
    const { results, model } = req.body;
    const selectedModel = model || 'anthropic/claude-3.5-sonnet';
    
    if (!results || !Array.isArray(results)) {
      return res.status(400).json({ 
        error: 'Invalid request: results array is required' 
      });
    }

    if (!OPENROUTER_API_KEY) {
      return res.status(500).json({ 
        error: 'OpenRouter API key not configured. Please set OPENROUTER_API_KEY in environment variables.',
        aiEnabled: false
      });
    }

    console.log(`Starting AI evaluation of ${results.length} approximations...`);
    
    const startTime = Date.now();
    const evaluatedResults = [];
    
    // Process approximations sequentially to avoid rate limiting
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      console.log(`AI evaluating approximation ${i + 1}/${results.length}...`);
      
      try {
        const aiEvaluation = await aiEvaluator.evaluateApproximation(result, selectedModel);
        evaluatedResults.push({
          ...result,
          aiEvaluation: aiEvaluation,
          evaluationIndex: i
        });
        
        // Add delay between requests to respect rate limits
        if (i < results.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
      } catch (error) {
        console.error(`Error evaluating approximation ${i + 1}:`, error);
        evaluatedResults.push({
          ...result,
          aiEvaluation: {
            accuracy: `Evaluation failed: ${error.message}`,
            efficiency: "Unable to analyze",
            novelty: "Unable to analyze",
            stability: "Unable to analyze", 
            generality: "Unable to analyze",
            recommendation: "Needs improvement"
          },
          evaluationIndex: i,
          evaluationError: error.message
        });
      }
    }
    
    const endTime = Date.now();
    console.log(`AI evaluation completed in ${(endTime - startTime) / 1000}s`);

    res.json({
      success: true,
      results: evaluatedResults,
      metadata: {
        totalApproximations: results.length,
        evaluationTimeMs: endTime - startTime,
        timestamp: new Date().toISOString(),
        model: selectedModel,
        provider: 'OpenRouter',
        aiEnabled: true
      }
    });

  } catch (error) {
    console.error('AI evaluation endpoint error:', error);
    res.status(500).json({ 
      error: 'Internal server error during AI evaluation',
      details: error.message,
      aiEnabled: !!OPENROUTER_API_KEY
    });
  }
});

// Test OpenRouter connection
app.get('/api/test-ai', async (req, res) => {
  if (!OPENROUTER_API_KEY) {
    return res.status(500).json({ 
      error: 'OpenRouter API key not configured',
      aiEnabled: false
    });
  }

  try {
    const response = await fetch(OPENROUTER_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://ramajan-evaluator.local',
        'X-Title': 'Ramanujan Mathematical Evaluator'
      },
      body: JSON.stringify({
        model: 'anthropic/claude-3.5-sonnet',
        messages: [
          {
            role: 'user',
            content: 'Respond with "OpenRouter connection successful" if you can read this.'
          }
        ],
        max_tokens: 50
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`API error: ${response.status} - ${errorData.error?.message || 'Unknown error'}`);
    }

    const data = await response.json();
    
    res.json({
      success: true,
      message: 'OpenRouter AI connection successful',
      response: data.choices[0]?.message?.content,
      model: data.model,
      usage: data.usage,
      aiEnabled: true
    });

  } catch (error) {
    console.error('OpenRouter test error:', error);
    res.status(500).json({ 
      error: 'Failed to connect to OpenRouter API',
      details: error.message,
      aiEnabled: false
    });
  }
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
║  ██████╗  █████╗ ███╗   ███╗ █████╗ ███╗   ██╗██╗   ██╗     ██╗ █████╗ ███╗   ██╗  ║
║  ██╔══██╗██╔══██╗████╗ ████║██╔══██╗████╗  ██║██║   ██║     ██║██╔══██╗████╗  ██║  ║
║  ██████╔╝███████║██╔████╔██║███████║██╔██╗ ██║██║   ██║     ██║███████║██╔██╗ ██║  ║
║  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══██║██║╚██╗██║██║   ██║██   ██║██╔══██║██║╚██╗██║  ║
║  ██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║██║ ╚████║╚██████╔╝╚█████╔╝██║  ██║██║ ╚████║  ║
║  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝  ║
║                                                                              ║
║    Mathematical Approximation Evaluator                                     ║
║    Server running on port ${PORT}                                               ║
║    ASCII vibes activated ⚡                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
  `);
});

module.exports = app;
