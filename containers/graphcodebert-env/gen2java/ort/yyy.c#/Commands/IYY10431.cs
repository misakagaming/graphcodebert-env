namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: IYY10431_XML_GROUP_CREATE_S      Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:30
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class IYY10431 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10431_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10431_OA WOa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.MYY10431_IA Myy10431Ia;
    GEN.ORT.YYY.MYY10431_OA Myy10431Oa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // REPEATING GROUP VIEW STATUS FIELDS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool ImpGroup_FL_001;
    int ImpGroup_PS_001;
    bool ImpGroup_RF_001;
    public const int ImpGroup_MM_001 = 10;
    bool ImpGroup_FL_002;
    int ImpGroup_PS_002;
    bool ImpGroup_RF_002;
    public const int ImpGroup_MM_002 = 10;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020173_esc_flag;
    //       +->   IYY10431_XML_GROUP_CREATE_S       01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Group View (10) imp_group
    //       !           Entity View imp_g iyy1_type (Transient, Optional,
    //       !                       Import only)
    //       !             tinstance_id
    //       !             treference_id
    //       !             tcreate_user_id
    //       !             tupdate_user_id
    //       !             tkey_attr_text
    //       !             tsearch_attr_text
    //       !             tother_attr_text
    //       !             tother_attr_date
    //       !             tother_attr_time
    //       !             tother_attr_amount
    //       !       EXPORTS:
    //       !         Work View exp canam_xml (Transient, Export only)
    //       !           xml_buffer
    //       !         Work View exp_error canam_xml (Transient, Export only)
    //       !           xml_return_code
    //       !           xml_message
    //       !           xml_position
    //       !           xml_source
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !   
    //     2 !   
    //     3 !  NOTE: 
    //     3 !  PURPOSE(CONTINUED)
    //     3 !  
    //     4 !  NOTE: 
    //     4 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     4 !  !!!!!!!!!!!!
    //     4 !  Check for Pre-Post Condition and Return/Reason Code
    //     4 !  
    //     5 !  NOTE: 
    //     5 !  PRE-CONDITION
    //     5 !  Give the attributes that will be composed as XML
    //     5 !  POST-CONDITION
    //     5 !  XML will be created
    //     5 !  Return Code = 1, Reason Code = 1
    //     5 !  
    //     6 !  NOTE: 
    //     6 !  PRE-CONDITION
    //     6 !  XML Create Error
    //     6 !  POST-CONDITION
    //     6 !  XML could not created. Check Exp_error canam_xml for details
    //     6 !  Return Code = -70, Reason Code = 1
    //     6 !  
    //     7 !  NOTE: 
    //     7 !  RETURN / REASON CODES
    //     7 !  +1/1 XML Created
    //     7 !  -70/1 XML Create error
    //     7 !  
    //     8 !  NOTE: 
    //     8 !  RELEASE HISTORY
    //     8 !  01_00 01-10-2009 New Release
    //     8 !  
    //     9 !  NOTE: 
    //     9 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     9 !  !!!!!!!!!!!!
    //     9 !  USE <mapper ab>
    //     9 !  
    //    10 !  USE myy10431_xml_group_create
    //    10 !     WHICH IMPORTS: Group View  imp_group TO Group View
    //    10 !              imp_group
    //    10 !     WHICH EXPORTS: Work View exp canam_xml FROM Work View exp
    //    10 !              canam_xml
    //    10 !                    Work View exp_error canam_xml FROM Work View
    //    10 !              exp_error canam_xml
    //    10 !                    Work View exp_error iyy1_component FROM Work
    //    10 !              View exp_error iyy1_component
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public IYY10431(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:30";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "IYY10431_XML_GROUP_CREATE_S";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	IYY10431_IA import_view, 
    	IYY10431_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      ++(NestingLevel);
      try {
        f_22020173_init(  );
        f_22020173(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020173(  )
    {
      func_0022020173_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020173" );
      Globdata.GetStateData().SetCurrentABName( "IYY10431_XML_GROUP_CREATE_S" );
      
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PURPOSE(CONTINUED)                                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    Check for Pre-Post Condition and Return/Reason Code             
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    Give the attributes that will be composed as XML                
      //    POST-CONDITION                                                  
      //    XML will be created                                             
      //    Return Code = 1, Reason Code = 1                                
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    XML Create Error                                                
      //    POST-CONDITION                                                  
      //    XML could not created. Check Exp_error canam_xml for details    
      //    Return Code = -70, Reason Code = 1                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RETURN / REASON CODES                                           
      //    +1/1 XML Created                                                
      //    -70/1 XML Create error                                          
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 01-10-2009 New Release                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    USE <mapper ab>                                                 
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      
      Myy10431Ia = (GEN.ORT.YYY.MYY10431_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10431).Assembly,
      	"GEN.ORT.YYY.MYY10431_IA" ));
      Myy10431Oa = (GEN.ORT.YYY.MYY10431_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10431).Assembly,
      	"GEN.ORT.YYY.MYY10431_OA" ));
      Myy10431Ia.ImpGroup_MA = IntAttr.ValueOf(WIa.ImpGroup_MA);
      for(Adim1 = 1; Adim1 <= WIa.ImpGroup_MA; ++(Adim1))
      {
        Myy10431Ia.ImpGIyy1TypeTinstanceId[Adim1-1] = TimestampAttr.ValueOf(WIa.ImpGIyy1TypeTinstanceId[Adim1-1]);
        Myy10431Ia.ImpGIyy1TypeTreferenceId[Adim1-1] = TimestampAttr.ValueOf(WIa.ImpGIyy1TypeTreferenceId[Adim1-1]);
        Myy10431Ia.ImpGIyy1TypeTcreateUserId[Adim1-1] = FixedStringAttr.ValueOf(WIa.ImpGIyy1TypeTcreateUserId[Adim1-1], 8);
        Myy10431Ia.ImpGIyy1TypeTupdateUserId[Adim1-1] = FixedStringAttr.ValueOf(WIa.ImpGIyy1TypeTupdateUserId[Adim1-1], 8);
        Myy10431Ia.ImpGIyy1TypeTkeyAttrText[Adim1-1] = FixedStringAttr.ValueOf(WIa.ImpGIyy1TypeTkeyAttrText[Adim1-1], 4);
        Myy10431Ia.ImpGIyy1TypeTsearchAttrText[Adim1-1] = FixedStringAttr.ValueOf(WIa.ImpGIyy1TypeTsearchAttrText[Adim1-1], 20);
        Myy10431Ia.ImpGIyy1TypeTotherAttrText[Adim1-1] = FixedStringAttr.ValueOf(WIa.ImpGIyy1TypeTotherAttrText[Adim1-1], 2);
        Myy10431Ia.ImpGIyy1TypeTotherAttrDate[Adim1-1] = DateAttr.ValueOf(WIa.ImpGIyy1TypeTotherAttrDate[Adim1-1]);
        Myy10431Ia.ImpGIyy1TypeTotherAttrTime[Adim1-1] = TimeAttr.ValueOf(WIa.ImpGIyy1TypeTotherAttrTime[Adim1-1]);
        Myy10431Ia.ImpGIyy1TypeTotherAttrAmount[Adim1-1] = DecimalAttr.ValueOf(WIa.ImpGIyy1TypeTotherAttrAmount[Adim1-1]);
      }
      for(Adim1 = WIa.ImpGroup_MA + 1; Adim1 <= 10; ++(Adim1))
      {
        Myy10431Ia.ImpGIyy1TypeTinstanceId[Adim1-1] = "00000000000000000000";
        Myy10431Ia.ImpGIyy1TypeTreferenceId[Adim1-1] = "00000000000000000000";
        Myy10431Ia.ImpGIyy1TypeTcreateUserId[Adim1-1] = "        ";
        Myy10431Ia.ImpGIyy1TypeTupdateUserId[Adim1-1] = "        ";
        Myy10431Ia.ImpGIyy1TypeTkeyAttrText[Adim1-1] = "    ";
        Myy10431Ia.ImpGIyy1TypeTsearchAttrText[Adim1-1] = "                    ";
        Myy10431Ia.ImpGIyy1TypeTotherAttrText[Adim1-1] = "  ";
        Myy10431Ia.ImpGIyy1TypeTotherAttrDate[Adim1-1] = 00000000;
        Myy10431Ia.ImpGIyy1TypeTotherAttrTime[Adim1-1] = 00000000;
        Myy10431Ia.ImpGIyy1TypeTotherAttrAmount[Adim1-1] = DecimalAttr.GetDefaultValue();
      }
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.MYY10431).Assembly,
      	"GEN.ORT.YYY.MYY10431",
      	"Execute",
      	Myy10431Ia,
      	Myy10431Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020173" );
      Globdata.GetStateData().SetCurrentABName( "IYY10431_XML_GROUP_CREATE_S" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Myy10431Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Myy10431Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Myy10431Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Myy10431Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Myy10431Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Myy10431Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Myy10431Oa.ExpErrorIyy1ComponentChecksum, 15);
      WOa.ExpErrorCanamXmlXmlReturnCode = FixedStringAttr.ValueOf(Myy10431Oa.ExpErrorCanamXmlXmlReturnCode, 2);
      WOa.ExpErrorCanamXmlXmlMessage = FixedStringAttr.ValueOf(Myy10431Oa.ExpErrorCanamXmlXmlMessage, 80);
      WOa.ExpErrorCanamXmlXmlPosition = DoubleAttr.ValueOf(Myy10431Oa.ExpErrorCanamXmlXmlPosition);
      WOa.ExpErrorCanamXmlXmlSource = FixedStringAttr.ValueOf(Myy10431Oa.ExpErrorCanamXmlXmlSource, 120);
      WOa.ExpCanamXmlXmlBuffer = StringAttr.ValueOf(Myy10431Oa.ExpCanamXmlXmlBuffer, 4094);
      Myy10431Ia.FreeInstance(  );
      Myy10431Ia = null;
      Myy10431Oa.FreeInstance(  );
      Myy10431Oa = null;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020173_init(  )
    {
      if ( NestingLevel < 2 )
      {
      }
      WOa.ExpCanamXmlXmlBuffer = "";
      WOa.ExpErrorCanamXmlXmlReturnCode = "  ";
      WOa.ExpErrorCanamXmlXmlMessage = "                                                                                ";
      WOa.ExpErrorCanamXmlXmlPosition = 0.0;
      WOa.ExpErrorCanamXmlXmlSource = 
        "                                                                                                                        ";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
      for(ImpGroup_PS_001 = 1; ImpGroup_PS_001 <= 10; ++(ImpGroup_PS_001))
      {
      }
      ImpGroup_PS_001 = 1;
      ImpGroup_PS_002 = 1;
    }
  }// end class
  
}

